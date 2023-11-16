<!--intro-start-->

# Holistic Evaluation of Text-To-Image Models

<img src="https://github.com/stanford-crfm/helm/raw/heim/src/helm/benchmark/static/heim/images/heim-logo.png" alt=""  width="800"/>

Significant effort has recently been made in developing text-to-image generation models, which take textual prompts as 
input and generate images. As these models are widely used in real-world applications, there is an urgent need to 
comprehensively understand their capabilities and risks. However, existing evaluations primarily focus on image-text 
alignment and image quality. To address this limitation, we introduce a new benchmark, 
**Holistic Evaluation of Text-To-Image Models (HEIM)**.

We identify 12 different aspects that are important in real-world model deployment, including:

- image-text alignment
- image quality
- aesthetics
- originality
- reasoning
- knowledge
- bias
- toxicity
- fairness
- robustness
- multilinguality
- efficiency

By curating scenarios encompassing these aspects, we evaluate state-of-the-art text-to-image models using this benchmark. 
Unlike previous evaluations that focused on alignment and quality, HEIM significantly improves coverage by evaluating all 
models across all aspects. Our results reveal that no single model excels in all aspects, with different models 
demonstrating strengths in different aspects.

<!--intro-end-->

**This repository contains the code used to produce the [results on the website](https://crfm.stanford.edu/heim/latest/) 
and paper. To get started, refer to the [documentation](https://crfm-heim.readthedocs.io/).**

 
