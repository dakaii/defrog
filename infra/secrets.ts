import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import * as k8s from "@pulumi/kubernetes";

interface SecretConfig {
    project: string;
    environment: string;
    k8sProvider: k8s.Provider;
    namespace: k8s.core.v1.Namespace;
}

export function createSecrets(config: SecretConfig) {
    const { project, environment, k8sProvider, namespace } = config;
    const prefix = `defrog-${environment}`;

    // Get sensitive configuration from Pulumi config
    const pulumiConfig = new pulumi.Config();
    const openaiApiKey = pulumiConfig.requireSecret("openai-api-key");
    const dbPassword = pulumiConfig.requireSecret("db-password");

    // Create Google Secret Manager secrets
    const secrets = {
        // OpenAI API Key
        openaiKey: new gcp.secretmanager.Secret(`${prefix}-openai-key`, {
            secretId: `${prefix}-openai-api-key`,
            replication: {
                automatic: {},
            },
            labels: {
                environment: environment,
                app: "defrog",
            },
        }),

        // Database Password
        dbPassword: new gcp.secretmanager.Secret(`${prefix}-db-password`, {
            secretId: `${prefix}-db-password`,
            replication: {
                automatic: {},
            },
            labels: {
                environment: environment,
                app: "defrog",
            },
        }),

        // JWT Secret for API
        jwtSecret: new gcp.secretmanager.Secret(`${prefix}-jwt-secret`, {
            secretId: `${prefix}-jwt-secret`,
            replication: {
                automatic: {},
            },
            labels: {
                environment: environment,
                app: "defrog",
            },
        }),
    };

    // Create secret versions with actual values
    const secretVersions = {
        openaiKeyVersion: new gcp.secretmanager.SecretVersion(`${prefix}-openai-key-version`, {
            secret: secrets.openaiKey.id,
            secretData: openaiApiKey.apply(key => key),
        }),

        dbPasswordVersion: new gcp.secretmanager.SecretVersion(`${prefix}-db-password-version`, {
            secret: secrets.dbPassword.id,
            secretData: dbPassword.apply(pwd => pwd),
        }),

        jwtSecretVersion: new gcp.secretmanager.SecretVersion(`${prefix}-jwt-secret-version`, {
            secret: secrets.jwtSecret.id,
            secretData: pulumi.secret(generateRandomString(32)),
        }),
    };

    // Create External Secrets Operator (ESO) resources if using ESO
    // Otherwise, create K8s secrets directly
    const k8sSecrets = {
        // OpenAI secret for K8s
        openaiSecret: new k8s.core.v1.Secret("openai-secret", {
            metadata: {
                name: "openai-secret",
                namespace: namespace.metadata.name,
            },
            type: "Opaque",
            stringData: {
                "api-key": openaiApiKey.apply(key => key),
            },
        }, { provider: k8sProvider }),

        // Database secret for K8s
        dbSecret: new k8s.core.v1.Secret("db-secret", {
            metadata: {
                name: "db-secret",
                namespace: namespace.metadata.name,
            },
            type: "Opaque",
            stringData: {
                "password": dbPassword.apply(pwd => pwd),
            },
        }, { provider: k8sProvider }),

        // JWT secret for K8s
        jwtSecret: new k8s.core.v1.Secret("jwt-secret", {
            metadata: {
                name: "jwt-secret",
                namespace: namespace.metadata.name,
            },
            type: "Opaque",
            stringData: {
                "secret": generateRandomString(32),
            },
        }, { provider: k8sProvider }),
    };

    // Create SecretProviderClass for GCP Secret Manager integration (optional, for more advanced setups)
    const secretProviderClass = new k8s.apiextensions.CustomResource("secret-provider-class", {
        apiVersion: "secrets-store.csi.x-k8s.io/v1",
        kind: "SecretProviderClass",
        metadata: {
            name: "defrog-secrets",
            namespace: namespace.metadata.name,
        },
        spec: {
            provider: "gcp",
            parameters: {
                secrets: pulumi.interpolate`
                - resourceName: "projects/${project}/secrets/${prefix}-openai-api-key/versions/latest"
                  path: "openai-api-key"
                - resourceName: "projects/${project}/secrets/${prefix}-db-password/versions/latest"
                  path: "db-password"
                - resourceName: "projects/${project}/secrets/${prefix}-jwt-secret/versions/latest"
                  path: "jwt-secret"
                `,
            },
        },
    }, { provider: k8sProvider });

    return {
        secrets,
        secretVersions,
        k8sSecrets,
        secretProviderClass,
    };
}

// Helper function to generate random strings
function generateRandomString(length: number): string {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += charset.charAt(Math.floor(Math.random() * charset.length));
    }
    return result;
}

// Function to create secret accessor IAM bindings
export function createSecretAccessorBindings(
    secrets: Record<string, gcp.secretmanager.Secret>,
    serviceAccountEmail: pulumi.Output<string>,
    project: string
) {
    const bindings: gcp.secretmanager.SecretIamBinding[] = [];

    for (const [key, secret] of Object.entries(secrets)) {
        const binding = new gcp.secretmanager.SecretIamBinding(`${key}-accessor`, {
            secretId: secret.id,
            role: "roles/secretmanager.secretAccessor",
            members: [pulumi.interpolate`serviceAccount:${serviceAccountEmail}`],
        });
        bindings.push(binding);
    }

    return bindings;
}