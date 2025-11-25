import logging

logger = logging.getLogger(__name__)


def _load_kube_config():
    from kubernetes import config
    from kubernetes.client.configuration import Configuration

    kube_config = Configuration()

    try:
        config.load_incluster_config(client_configuration=kube_config)
        logger.info("Loaded in-cluster Kubernetes configuration")
    except config.ConfigException:
        config.load_kube_config(client_configuration=kube_config)
        logger.info("Loaded Kubernetes configuration from kubeconfig file")

    return kube_config