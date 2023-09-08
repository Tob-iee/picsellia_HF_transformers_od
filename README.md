# Integrate Picsellia With Hugginngface Transformers



## 📚 Table of Contents

* Overview
* Getting Started
    * Prerequisites
    * Installation
    * Setup and spin up a training experiment with Docker


## 📌 Overview

This repository provides a guide for integrating Picsellia into a Huggingface Transformer (object detection) training script.
Implementing computer vision model training at scale requires a robust [computer vision operations (CVOps)](https://www.picsellia.com/post/how-to-apply-mlops-to-computer-vision-cvops) workflow. Scaling training with a custom workflow can bottleneck at the point of traceability and repeatability. Tailoring this into your custom training workflow can be costly in terms of time and resources.

A CVOps platform like Picsellia has robust architectural workflows that provide enough features to handle large-scale computer vision-based model development (training). It has the necessary tools to manage the entire CV development lifecycle. You have two options when leveraging Picsellia for training. You can re-develop your entire training workflow to use Picsellia's workflows and infrastructure. Or integrate Picsellia into your existing training workflow and still use your infrastructure. The integration option gives you the best of both worlds without a platform lock or the compulsory mandate of migrating your existing training workflows to the platform. It also creates two points of failure for the training workflows and multiple data locations since the training information generated in your local training environment (on the infrastructure) also exists on Picsellia's platform.


## 🚀 Getting Started

### ✅ Prerequisites

 - An active Picsellia account.
 - The Picsellia SDK is installed on your machine.
 - HuggingFace is installed on your machine.
 - Picsellia's requirements for an integration; see requirements [here](https://documentation.picsellia.com/docs/part-1-overview-1).
 - Basic-to-intermediate understanding of experiment tracking.


### Installation ⚙️

All the dependencies you'll need are in the requirements.txt. To install, run:

``` pip install -r requirements.txt ```

### 🛠️️ Setup and spin up a training experiment with Docker

First and foremost, make sure that [Docker](https://www.docker.com/) is installed and working properly in the system.

For a detailed breakdown of the steps in the training workflow read this [Blog](https://www.dropbox.com/scl/fi/37s8725vmn8wmt2jp8ikm/How-to-Integrate-Picsellia-into-a-Hugging-Face-Training-Workflow.paper?rlkey=phquba1etuwwk84lp7vohn4fh&dl=0)


1. Clone the repository:

``` git clone git@github.com:Tob-iee/picsellia_HF_transformers_od.git ```






