# Monitoring Setup Guide

This guide explains how to set up and use the monitoring stack, which includes Prometheus, Grafana, and Loki.

## Overview

The monitoring stack is designed to give you a comprehensive overview of the health and performance of your media server. It consists of the following services:

*   **Prometheus:** Collects metrics from your services.
*   **Grafana:** Visualizes your metrics in beautiful dashboards.
*   **Loki:** Collects logs from your services.
*   **Promtail:** Ships logs to Loki.

## Initial Setup

The `deploy.sh` script automatically creates a basic configuration file for Prometheus at `config/prometheus/prometheus.yml`. This file is configured to scrape metrics from the Docker daemon and Prometheus itself.

If you deployed the stack manually, you will need to create this file yourself. You can use the following template:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9090']
        
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
```

## Accessing Grafana

Once the stack is deployed, you can access the Grafana dashboard at `http://localhost:3000`. The default credentials are:

*   **Username:** `admin`
*   **Password:** The value of `GRAFANA_PASSWORD` in your `.env` file.

## Exploring Metrics

Grafana comes with a pre-configured data source for Prometheus. You can start exploring your metrics by creating a new dashboard:

1.  Click the "+" icon in the sidebar and select "Dashboard".
2.  Click "Add new panel".
3.  In the "Metrics" field, enter a PromQL query. For example, to see the CPU usage of your containers, you can use the following query:

    ```promql
    sum(rate(container_cpu_usage_seconds_total[1m])) by (name)
    ```

4.  Choose a visualization type (e.g., "Graph") and customize the panel to your liking.

## Exploring Logs

Grafana also comes with a pre-configured data source for Loki. You can explore your logs in the "Explore" view:

1.  Click the "Explore" icon in the sidebar.
2.  Select the "Loki" data source from the dropdown menu.
3.  In the "Log browser" field, enter a LogQL query. For example, to see the logs for the Sonarr container, you can use the following query:

    ```logql
    {container="sonarr"}
    ```

4.  You can then filter and search the logs to find the information you need.

## Next Steps

This guide provides a basic overview of the monitoring stack. To learn more about how to use Prometheus, Grafana, and Loki, please refer to their official documentation:

*   [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
*   [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
*   [Loki Documentation](https://grafana.com/docs/loki/latest/)
