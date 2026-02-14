import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as k8s from "@pulumi/kubernetes";
import * as docker from "@pulumi/docker";

// Get configuration
const config = new pulumi.Config();
const gcpConfig = new pulumi.Config("gcp");
const project = gcpConfig.require("project");
const region = gcpConfig.require("region");
const zone = gcpConfig.require("zone");
const environment = config.require("environment");

// Resource naming prefix
const prefix = `defrog-${environment}`;

// Create VPC network
const network = new gcp.compute.Network(`${prefix}-network`, {
    name: `${prefix}-network`,
    autoCreateSubnetworks: false,
    description: "DeFrog VPC network",
});

// Create subnet
const subnet = new gcp.compute.Subnetwork(`${prefix}-subnet`, {
    name: `${prefix}-subnet`,
    network: network.id,
    region: region,
    ipCidrRange: "10.0.0.0/24",
    secondaryIpRanges: [
        {
            rangeName: "pods",
            ipCidrRange: "10.1.0.0/16",
        },
        {
            rangeName: "services",
            ipCidrRange: "10.2.0.0/16",
        },
    ],
});

// Create Cloud SQL instance with pgvector
const dbInstance = new gcp.sql.DatabaseInstance(`${prefix}-postgres`, {
    name: `${prefix}-postgres`,
    databaseVersion: "POSTGRES_15",
    region: region,
    deletionProtection: environment === "prod",
    settings: {
        tier: environment === "prod" ? "db-custom-2-8192" : "db-f1-micro",
        diskSize: environment === "prod" ? 100 : 10,
        diskType: "PD_SSD",
        backupConfiguration: {
            enabled: true,
            startTime: "03:00",
            pointInTimeRecoveryEnabled: environment === "prod",
            transactionLogRetentionDays: environment === "prod" ? 7 : 1,
        },
        ipConfiguration: {
            ipv4Enabled: true,
            requireSsl: environment === "prod",
            privateNetwork: network.selfLink,
            authorizedNetworks: environment === "dev" ? [
                {
                    name: "allow-all-dev",
                    value: "0.0.0.0/0",
                },
            ] : [],
        },
        databaseFlags: [
            {
                name: "cloudsql.enable_pgvector",
                value: "on",
            },
        ],
        insightsConfig: {
            queryInsightsEnabled: true,
            queryStringLength: 1024,
            recordApplicationTags: true,
            recordClientAddress: true,
        },
    },
});

// Create database
const database = new gcp.sql.Database(`${prefix}-database`, {
    name: "defrog",
    instance: dbInstance.name,
});

// Create database user
const dbUser = new gcp.sql.User(`${prefix}-db-user`, {
    name: "defrog",
    instance: dbInstance.name,
    password: pulumi.secret("defrog-db-password"), // Change this in production
});

// Create GKE cluster
const cluster = new gcp.container.Cluster(`${prefix}-gke`, {
    name: `${prefix}-gke`,
    location: region,
    deletionProtection: environment === "prod",
    removeDefaultNodePool: true,
    initialNodeCount: 1,
    network: network.name,
    subnetwork: subnet.name,
    ipAllocationPolicy: {
        clusterSecondaryRangeName: "pods",
        servicesSecondaryRangeName: "services",
    },
    masterAuth: {
        clientCertificateConfig: {
            issueClientCertificate: false,
        },
    },
    workloadIdentityConfig: {
        workloadPool: `${project}.svc.id.goog`,
    },
    addonsConfig: {
        horizontalPodAutoscaling: { disabled: false },
        httpLoadBalancing: { disabled: false },
        gcpFilestoreCsiDriverConfig: { enabled: true },
    },
});

// Create GKE node pool
const nodePool = new gcp.container.NodePool(`${prefix}-node-pool`, {
    name: `${prefix}-node-pool`,
    cluster: cluster.name,
    location: region,
    nodeCount: environment === "prod" ? 3 : 2,
    nodeConfig: {
        machineType: environment === "prod" ? "n2-standard-2" : "e2-medium",
        diskSizeGb: 50,
        diskType: "pd-standard",
        oauthScopes: [
            "https://www.googleapis.com/auth/cloud-platform",
        ],
        workloadMetadataConfig: {
            mode: "GKE_METADATA",
        },
        labels: {
            environment: environment,
            app: "defrog",
        },
    },
    autoscaling: {
        minNodeCount: environment === "prod" ? 2 : 1,
        maxNodeCount: environment === "prod" ? 10 : 3,
    },
    management: {
        autoRepair: true,
        autoUpgrade: true,
    },
});

// Create Artifact Registry for Docker images
const registry = new gcp.artifactregistry.Repository(`${prefix}-registry`, {
    repositoryId: `${prefix}-docker`,
    location: region,
    format: "DOCKER",
    description: "Docker registry for DeFrog application",
});

// Create Secret Manager secrets
const openaiApiKey = new gcp.secretmanager.Secret(`${prefix}-openai-key`, {
    secretId: `${prefix}-openai-key`,
    replication: {
        automatic: {},
    },
});

const openaiApiKeyVersion = new gcp.secretmanager.SecretVersion(`${prefix}-openai-key-version`, {
    secret: openaiApiKey.id,
    secretData: "YOUR_OPENAI_API_KEY", // Replace with actual key
});

// Create service account for workload identity
const serviceAccount = new gcp.serviceaccount.Account(`${prefix}-workload-sa`, {
    accountId: `${prefix}-workload-sa`,
    displayName: "DeFrog Workload Service Account",
});

// Grant necessary permissions
const sqlClientRole = new gcp.projects.IAMBinding(`${prefix}-sql-client`, {
    project: project,
    role: "roles/cloudsql.client",
    members: [pulumi.interpolate`serviceAccount:${serviceAccount.email}`],
});

const secretAccessorRole = new gcp.projects.IAMBinding(`${prefix}-secret-accessor`, {
    project: project,
    role: "roles/secretmanager.secretAccessor",
    members: [pulumi.interpolate`serviceAccount:${serviceAccount.email}`],
});

// Create K8s provider
const k8sProvider = new k8s.Provider(`${prefix}-k8s`, {
    kubeconfig: pulumi.all([cluster.name, cluster.endpoint, cluster.masterAuth]).apply(
        ([name, endpoint, masterAuth]) => {
            const context = `gke_${project}_${region}_${name}`;
            return `apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: ${masterAuth.clusterCaCertificate}
    server: https://${endpoint}
  name: ${context}
contexts:
- context:
    cluster: ${context}
    user: ${context}
  name: ${context}
current-context: ${context}
kind: Config
preferences: {}
users:
- name: ${context}
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1beta1
      command: gke-gcloud-auth-plugin
      installHint: Install gke-gcloud-auth-plugin for use with kubectl by following
        https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke
      provideClusterInfo: true`;
        }
    ),
});

// Create namespace
const namespace = new k8s.core.v1.Namespace(`${prefix}-namespace`, {
    metadata: {
        name: "defrog",
        labels: {
            environment: environment,
        },
    },
}, { provider: k8sProvider });

// Import K8s deployment function
import { createK8sDeployment } from "./k8s-deployment";

// Create K8s deployments
const k8sDeployments = createK8sDeployment({
    namespace,
    k8sProvider,
    dbInstanceConnectionName: dbInstance.connectionName,
    registryUrl: pulumi.interpolate`${region}-docker.pkg.dev/${project}/${registry.repositoryId}`,
    serviceAccountEmail: serviceAccount.email,
    environment,
    project,
});

// Export important values
export const networkName = network.name;
export const subnetName = subnet.name;
export const clusterName = cluster.name;
export const clusterEndpoint = cluster.endpoint;
export const dbInstanceConnectionName = dbInstance.connectionName;
export const dbInstanceIp = dbInstance.publicIpAddress;
export const registryUrl = pulumi.interpolate`${region}-docker.pkg.dev/${project}/${registry.repositoryId}`;
export const serviceAccountEmail = serviceAccount.email;
export const namespaceName = namespace.metadata.name;
export const dashboardUrl = k8sDeployments.streamlitService.status.loadBalancer.ingress[0].ip;