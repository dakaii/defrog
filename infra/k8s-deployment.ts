import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";
import * as gcp from "@pulumi/gcp";

interface DeploymentConfig {
    namespace: k8s.core.v1.Namespace;
    k8sProvider: k8s.Provider;
    dbInstanceConnectionName: pulumi.Output<string>;
    registryUrl: pulumi.Output<string>;
    serviceAccountEmail: pulumi.Output<string>;
    environment: string;
    project: string;
}

export function createK8sDeployment(config: DeploymentConfig) {
    const { namespace, k8sProvider, dbInstanceConnectionName, registryUrl, serviceAccountEmail, environment, project } = config;
    
    // Create ConfigMap for app configuration
    const appConfig = new k8s.core.v1.ConfigMap("defrog-config", {
        metadata: {
            name: "defrog-config",
            namespace: namespace.metadata.name,
        },
        data: {
            POSTGRES_HOST: "127.0.0.1",
            POSTGRES_PORT: "5432",
            POSTGRES_DB: "defrog",
            POSTGRES_USER: "defrog",
            LLM_MODEL: "gpt-4o-mini",
            EMBEDDING_MODEL: "text-embedding-3-small",
            API_URL: "http://defrog-api:8000",
            ENVIRONMENT: environment,
        },
    }, { provider: k8sProvider });

    // Create Secret for sensitive data (populated from Pulumi config)
    const pulumiConfig = new pulumi.Config();
    const appSecret = new k8s.core.v1.Secret("defrog-secret", {
        metadata: {
            name: "defrog-secret",
            namespace: namespace.metadata.name,
        },
        type: "Opaque",
        stringData: {
            POSTGRES_PASSWORD: pulumiConfig.requireSecret("db-password"),
        },
    }, { provider: k8sProvider });

    // Create service account for workload identity
    const k8sServiceAccount = new k8s.core.v1.ServiceAccount("defrog-ksa", {
        metadata: {
            name: "defrog-ksa",
            namespace: namespace.metadata.name,
            annotations: {
                "iam.gke.io/gcp-service-account": serviceAccountEmail,
            },
        },
    }, { provider: k8sProvider });

    // Bind GCP service account to K8s service account
    const workloadIdentityBinding = new gcp.serviceaccount.IAMBinding("defrog-workload-identity", {
        serviceAccountId: serviceAccountEmail,
        role: "roles/iam.workloadIdentityUser",
        members: [pulumi.interpolate`serviceAccount:${project}.svc.id.goog[${namespace.metadata.name}/defrog-ksa]`],
    });

    // Create API deployment
    const apiDeployment = new k8s.apps.v1.Deployment("defrog-api", {
        metadata: {
            name: "defrog-api",
            namespace: namespace.metadata.name,
            labels: {
                app: "defrog-api",
                environment: environment,
            },
        },
        spec: {
            replicas: environment === "prod" ? 3 : 1,
            selector: {
                matchLabels: {
                    app: "defrog-api",
                },
            },
            template: {
                metadata: {
                    labels: {
                        app: "defrog-api",
                        environment: environment,
                    },
                },
                spec: {
                    serviceAccountName: k8sServiceAccount.metadata.name,
                    containers: [
                        {
                            name: "api",
                            image: pulumi.interpolate`${registryUrl}/defrog-api:latest`,
                            imagePullPolicy: "Always",
                            ports: [
                                {
                                    containerPort: 8000,
                                    name: "http",
                                },
                            ],
                            envFrom: [
                                {
                                    configMapRef: {
                                        name: appConfig.metadata.name,
                                    },
                                },
                                {
                                    secretRef: {
                                        name: appSecret.metadata.name,
                                    },
                                },
                            ],
                            env: [
                                {
                                    name: "OPENAI_API_KEY",
                                    valueFrom: {
                                        secretKeyRef: {
                                            name: "openai-secret",
                                            key: "api-key",
                                        },
                                    },
                                },
                            ],
                            resources: {
                                requests: {
                                    cpu: "100m",
                                    memory: "256Mi",
                                },
                                limits: {
                                    cpu: environment === "prod" ? "1000m" : "500m",
                                    memory: environment === "prod" ? "1Gi" : "512Mi",
                                },
                            },
                            livenessProbe: {
                                httpGet: {
                                    path: "/health",
                                    port: 8000,
                                },
                                initialDelaySeconds: 30,
                                periodSeconds: 10,
                            },
                            readinessProbe: {
                                httpGet: {
                                    path: "/health",
                                    port: 8000,
                                },
                                initialDelaySeconds: 10,
                                periodSeconds: 5,
                            },
                        },
                        {
                            name: "cloud-sql-proxy",
                            image: "gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.8.0",
                            args: [
                                pulumi.interpolate`--private-ip`,
                                pulumi.interpolate`${dbInstanceConnectionName}`,
                                "--port=5432",
                                "--auto-iam-authn",
                            ],
                            resources: {
                                requests: {
                                    cpu: "50m",
                                    memory: "64Mi",
                                },
                                limits: {
                                    cpu: "100m",
                                    memory: "128Mi",
                                },
                            },
                        },
                    ],
                },
            },
        },
    }, { provider: k8sProvider });

    // Create API service
    const apiService = new k8s.core.v1.Service("defrog-api-service", {
        metadata: {
            name: "defrog-api",
            namespace: namespace.metadata.name,
            labels: {
                app: "defrog-api",
            },
        },
        spec: {
            type: "ClusterIP",
            selector: {
                app: "defrog-api",
            },
            ports: [
                {
                    port: 8000,
                    targetPort: 8000,
                    protocol: "TCP",
                    name: "http",
                },
            ],
        },
    }, { provider: k8sProvider });

    // Create Streamlit deployment
    const streamlitDeployment = new k8s.apps.v1.Deployment("defrog-dashboard", {
        metadata: {
            name: "defrog-dashboard",
            namespace: namespace.metadata.name,
            labels: {
                app: "defrog-dashboard",
                environment: environment,
            },
        },
        spec: {
            replicas: environment === "prod" ? 2 : 1,
            selector: {
                matchLabels: {
                    app: "defrog-dashboard",
                },
            },
            template: {
                metadata: {
                    labels: {
                        app: "defrog-dashboard",
                        environment: environment,
                    },
                },
                spec: {
                    serviceAccountName: k8sServiceAccount.metadata.name,
                    containers: [
                        {
                            name: "dashboard",
                            image: pulumi.interpolate`${registryUrl}/defrog-dashboard:latest`,
                            imagePullPolicy: "Always",
                            ports: [
                                {
                                    containerPort: 8501,
                                    name: "http",
                                },
                            ],
                            env: [
                                {
                                    name: "API_URL",
                                    value: "http://defrog-api:8000",
                                },
                            ],
                            resources: {
                                requests: {
                                    cpu: "100m",
                                    memory: "256Mi",
                                },
                                limits: {
                                    cpu: environment === "prod" ? "500m" : "250m",
                                    memory: environment === "prod" ? "512Mi" : "384Mi",
                                },
                            },
                            livenessProbe: {
                                httpGet: {
                                    path: "/",
                                    port: 8501,
                                },
                                initialDelaySeconds: 30,
                                periodSeconds: 10,
                            },
                        },
                    ],
                },
            },
        },
    }, { provider: k8sProvider });

    // Create Streamlit service
    const streamlitService = new k8s.core.v1.Service("defrog-dashboard-service", {
        metadata: {
            name: "defrog-dashboard",
            namespace: namespace.metadata.name,
            labels: {
                app: "defrog-dashboard",
            },
        },
        spec: {
            type: "LoadBalancer",
            selector: {
                app: "defrog-dashboard",
            },
            ports: [
                {
                    port: 80,
                    targetPort: 8501,
                    protocol: "TCP",
                    name: "http",
                },
            ],
        },
    }, { provider: k8sProvider });

    // Create HPA for API
    const apiHPA = new k8s.autoscaling.v2.HorizontalPodAutoscaler("defrog-api-hpa", {
        metadata: {
            name: "defrog-api-hpa",
            namespace: namespace.metadata.name,
        },
        spec: {
            scaleTargetRef: {
                apiVersion: "apps/v1",
                kind: "Deployment",
                name: apiDeployment.metadata.name,
            },
            minReplicas: environment === "prod" ? 2 : 1,
            maxReplicas: environment === "prod" ? 10 : 3,
            metrics: [
                {
                    type: "Resource",
                    resource: {
                        name: "cpu",
                        target: {
                            type: "Utilization",
                            averageUtilization: 70,
                        },
                    },
                },
                {
                    type: "Resource",
                    resource: {
                        name: "memory",
                        target: {
                            type: "Utilization",
                            averageUtilization: 80,
                        },
                    },
                },
            ],
        },
    }, { provider: k8sProvider });

    return {
        apiDeployment,
        apiService,
        streamlitDeployment,
        streamlitService,
        apiHPA,
        appConfig,
        appSecret,
        k8sServiceAccount,
    };
}