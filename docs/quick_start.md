# Quick Start

To run HEIM, follow these steps:

1. Create a run specs configuration file. For example, to evaluate 
[Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) against the 
[MS-COCO scenario](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/scenarios/image_generation/mscoco_scenario.py), run:
```
echo 'entries: [{description: "mscoco:model=huggingface/stable-diffusion-v1-4", priority: 1}]' > run_specs.conf
```
2. Run the benchmark with certain number of instances (e.g., 10 instances): 
`helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 10`
3. Summarize and aggregate the results: `helm-summarize --suite v1`.
4. Start a web server to display benchmark results: `helm-server`
5. Open http://localhost:8000/ in your favorite browser to see the results and generated images.

Examples of run specs configuration files can be found [here](https://github.com/stanford-crfm/heim/tree/main/src/helm/benchmark/presentation).
We used [this configuration file](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/presentation/run_specs_heim.conf) 
to produce results of the paper.


## Perspective API

We used Google's [Perspective API](https://www.perspectiveapi.com) to calculate the toxicity of textual prompts
(e.g., [NSFWMetric](https://github.com/stanford-crfm/heim/blob/main/src/helm/benchmark/metrics/image_generation/nsfw_metrics.py#L29)).
To send requests to PerspectiveAPI, we need to generate an API key from GCP. Follow the
[Get Started guide](https://developers.perspectiveapi.com/s/docs-get-started)
to request the service and the [Enable the API guide](https://developers.perspectiveapi.com/s/docs-enable-the-api)
to generate the API key. Once you have a valid API key, add an entry to `prod_env/credentials.conf`:

```
perspectiveApiKey: <Generated API key>
```

By default, Perspective API allows only 1 query per second. Fill out this
[form](https://developers.perspectiveapi.com/s/request-quota-increase) to increase the request quota.
