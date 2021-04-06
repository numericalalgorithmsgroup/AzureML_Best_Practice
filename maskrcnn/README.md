![NAG Logo](https://www.nag.com/themes/custom/nag/logo.png)

# Tutorial: Training at Scale on AzureML

*Learn how to quickly train an AI model at scale using Azure Machine Learning.*

**Note: The cloud is a fast-moving environment and things can change quickly.  This tutorial is
accurate as of March 2021 but you should check the [Azure Machine Learning
Documentation](https://docs.microsoft.com/en-gb/azure/machine-learning/) for the latest updates to
the service.**

AI and Machine learning are transforming science, industry and business with an always expanding
range of applications. The pace of progress is relentless and with models becoming ever more
complex and datasets ever larger, a single GPU or even a single machine with multiple GPUs is often
not enough. Distributed training on large GPU clusters is becoming a more common requirement. For
many organizations, owning such a cluster is not the best solution, so the cloud is a natural way
to access large GPU clusters. In this tutorial we will show you how to quickly train a distributed
model on your own GPU cluster using Azure Machine Learning.

It's no secret that cloud computing can be complex, especially when directly managing
infrastructure such as VMs and virtual networks. However, by using appropriate managed services the
underlying infrastructure management is handled by the cloud platform.  The Azure Machine Learning
service allows the user to manage all aspects of the training (or inference) being performed, while
automatically managing the underlying infrastructure.

By the end of this tutorial, you will understand how to create an AzureML workspace and configure
datasets, software environments and compute resources. You will also learn how to create and submit
training jobs using the AzureML Python SDK.

We will use Mask R-CNN as an example of a large model which scales well to many GPUs on multiple
nodes.  Mask R-CNN is an image segmentation and object detection model which is designed to
identify all objects in a provided image and create pixel-level masks for each detected object.
The implementation we have chosen is available from the NVIDIA [Deep Learning Examples] repository
on Github.  The training dataset we will use is the COCO2017 (Common Objects in Context) dataset
which is used for the MLPerf benchmark of Mask R-CNN.  Below is an example of the output that is 
attainable from a well-trained Mask R-CNN implementation:

![Image Credit: Facebook AI Research (https://github.com/facebookresearch/detectron2/)](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)
[Image Credit: Facebook AI Research (https://github.com/facebookresearch/detectron2/)](https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png)


[Deep Learning Examples]:https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/MaskRCNN
[COCO2017]: https://cocodataset.org

### Prerequisites

To follow this tutorial you will need:

* An Azure subscription with ND40rs v2 instance quota for AzureML
* Basic familiarity with the Linux Bash shell
* Basic familiarity with the Python programming language


## Preparing Datasets and Infrastructure

Before we can do any training we must first create and configure the various Azure resources we
will be using.  First, we will create a storage account and upload the training dataset, then we can
create an AzureML workspace and configure it to allow access to the dataset for training.

These first steps will be performed using the [Azure CLI] tool and Bash shell. The Azure CLI can be
installed on Windows, Linux and Mac OS, or is available via the [Azure cloud shell]. The Bash shell
is available on Linux and Mac OS, or on Windows via the [Windows Subsystem for Linux] (WSL).
Alternatively, all steps that require the Bash shell can also be achieved using an Azure VM instance
running a Linux distribution (e.g Ubuntu 18.04).

[Azure CLI]: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
[Azure cloud shell]: https://docs.microsoft.com/en-us/azure/cloud-shell/quickstart
[Windows Subsystem for Linux]: https://docs.microsoft.com/en-us/windows/wsl/install-win10]

Note: This tutorial assumes that you do not have any existing AzureML workspaces or other
infrastructure provisioned and so goes through all the creation steps needed. However, if you have
already created an AzureML workspace or suitable storage accounts you may wish to use them rather
than create new ones.

### Preparing the Training Dataset

AzureML supports accessing data from a variety of Azure storage options. However for large
file-based datasets, such as the [COCO2017] dataset we will be using in this example, the most
appropriate storage backend is [Azure blob storage].

[Azure blob storage]: https://docs.microsoft.com/en-gb/azure/storage/blobs/storage-blobs-introduction


To use Azure blob as our storage backend we must first create a [storage account] and
[blob container].  We can, of course, use any existing storage account for this, but for best
performance, we recommended creating a new dedicated storage account for training datasets.  This
avoids sharing bandwidth and transaction caps with other workloads which could degrade training
performance and increase compute costs.  _It is also vital that the storage account is in the same
region as the AzureML workspace.  This ensures maximum performance as well as avoiding costs for
data transfer between regions._

[storage account]: https://docs.microsoft.com/en-gb/azure/storage/common/storage-account-create
[blob container]:  https://docs.microsoft.com/en-gb/azure/storage/blobs/storage-blobs-introduction

You can create a storage account and container via the Azure Portal or the Azure CLI. For this
tutorial we will demonstrate using the Azure CLI:

```bash
$ location="WestEurope"
$ rg_name="MLRG"
$ az group create --name ${rg_name} --location ${location}
$ storage_name=mlstorage-$RANDOM
$ az storage account create --name ${storage_name} --resource-group ${rg_name} --location ${location} --sku Standard_LRS
$ container_name=coco2017
```
In all the examples in this tutorial we create shell variables to hold names and other
configuration options. These variables are then used in subsequent examples to refer to previously
created resource.  You should modify the names of resources and other options as appropriate to 
meet your needs.

Once you have created the storage account and container you should also obtain a [shared access
token](https://docs.microsoft.com/en-us/azure/storage/common/storage-sas-overview) (SAS) which can
be used to grant AzureML access to the container.  The following Azure CLI commands will create a
temporary (1 month lifespan) SAS token with the appropriate permissions and stores it in the
`${sas}` shell variable for future use in this tutorial:

```bash
$ expiry=$(date -u -d "1 month" '+%Y-%m-%dT%H:%MZ')
$ sas=$(az storage account generate-sas --account-name $storage_name --expiry $expiry \
  --services b --resource-types co --permission acdlpruw -o tsv)
```


Uploading the dataset can be done in various ways, including using the Azure Portal, [Storage
Explorer] or using the [`azcopy`] command-line tool.  Using `azcopy` is the most flexible as it can
be used with local files, Azure storage and various other cloud vendor storage technologies to
manage upload to, download from and transfer between cloud storage accounts.

[Storage Explorer]: https://azure.microsoft.com/en-us/features/storage-explorer/
[`azcopy`]: https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10

If you are following this tutorial using Mask RCNN you can download and prepare the COCO 2017 dataset with
the following commands:

```bash
$ mkdir cocotmp; cd cocotmp
$ wget http://images.cocodataset.org/zips/train2017.zip
$ unzip train2017.zip
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
```
If you are downloading a dataset from the internet we recommend that you use an Azure Virtual
Machine in the same region as the storage account to prepare the data.  This avoids having to
download and then re-upload the data over your local internet connection.

Once you have prepared your dataset it must be uploaded to Azure storage. To upload the prepared
dataset to your chosen Azure storage container, make sure you are in the base directory for the
dataset, then use the following `azcopy` command:
```bash
$ azcopy copy -r . https://${storage_name}.blob.core.windows.net/${container_name}/?${sas}
```
This command will recursively copy the contents of the current working directory and all
subdirectories to the previously created Azure storage container.

### Create an AzureML Workspace

The next step will be to create an AzureML workspace.  This is a logical container
or account that all of our machine learning configuration, data, experiments and outputs will be
stored in.  You can create an AzureML workspace either via the Azure portal, the
Azure CLI or the AzureML Python SDK.

The workspace depends on various other Azure resources including a KeyVault, storage account, and
container registry.  By default new resources will be created, but you can optionally choose to use
existing resources and grant the Workspace access to them. In this tutorial, we will not provide
specific resources and so a new Keyvault, storage account and container registry will be
automatically created along with the workspace.

To create a new workspace from the Azure CLI, use the following command:

```bash
$ azml_workspace="MLWorkspace"
$ az ml workspace create --resource-group ${rg_name} \
  --workspace_name ${azml_workspace} \
  --location ${location}
```
This will typically take several minutes as all the dependent resources must also be created.

Once the workspace and it's dependent resources are created we can begin configuring it with our
datasets and training environment.

### Registering the Training Dataset

To make the training dataset available for our scripts to consume inside the AzureML training
environment we need to register the dataset and the storage backend in AzureML.  To do this the
AzureML SDK provides [`Datastore`] and [`Dataset`] objects which define the storage backend
configuration and dataset configuration respectively.

[`Datastore`]: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data
[`Dataset`]: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets

A `Datastore` object defines a connection to a specific storage backend and securely caches any
required authentication information.  In this case, we are connecting to the Azure Blob storage
backend and authentication using a shared access signature (SAS).  This can be done via the Machine
Learning Studio, Azure CLI or the AzureML SDK.

To define a `Datastore` that attaches to our Azure Blob storage via the Azure CLI, use the following commmand:

```bash
$ datastore_name="coco2017_blob"
$ az ml datastore attach-blob \
  --resource-group ${rg_name} \
  --workspace-name ${azml_workspace} \
  --name ${datastore_name} \
  --account-name ${storage_name} \
  --container-name ${container_name} \
  --sas-token ${sas}
```
(Note: this assumes you have performed the previous steps to set the workspace name, storage
account name, container name and shared access signature shell variables.)

We can now define a `DataSet` using this `Datastore`.  Specifically, we will create a `FileDataSet`
object, which defines a list of files from the storage which can be mounted or downloaded when we
perform a training or inference run in the AzureML workspace.  We can provide an explicit list of
all the files in the dataset or provide a pattern using wildcards ('\*'). In this case, since the
container contains only the dataset files and nothing else, we can use the pattern '/\*\*' to
match all files and folders in the container recursively. Again, this can be done via the Machine
Learning Studio, Azure CLI, or the AzureML SDK.

To define a `FileDataSet` for the COCO 2017 training set via the Azure CLI, first create a DataSet
specification file containing the following:

```json
{
  "datasetType": "File",
  "parameters": {
    "path": {
        "datastoreName": "coco2017_blob",
        "relativePath": "/**"
    }
  },
  "registration": {
    "description": "COCO 2017 training and validation images",
    "name": "coco2017_trainval"
  },
  "schemaVersion": 1
}
```
If you are using a different dataset or naming scheme ensure you substitute the appropriate value
of `dataStoreName` and `name`, then save the file in your working directory as e.g
`dataset_spec.json`.

Now we can use the following command to register a `Dataset` object in our workspace with the spec
file:

```bash
$ dataset_name="coco2017_trainval"
$ az ml dataset register --workspace-name ${azml_workspace} \
  --resource-group ${rg_name} \
  --file dataset_spec.json
```
This `Dataset` object can now be used in any AzureML jobs to access the dataset files we prepared
and uploaded.

## Submission Scripts

The rest of this tutorial is concerned with configuring the compute environment and job
configuration for individual training runs. This is typically unique to specific runs which may
require a different software environment, number of compute nodes or type of compute node, as well
as specific framework and model configuration.

We have provided example job submission scripts and their accompanying configuration files in the
[AzureML_Best_Practice] Github repository. To obtain them you should download or clone this repository.
The example Mask R-CNN training benchmark scripts can then be found in the `maskrcnn` subdirectory.

[AzureML_Best_Practice]: https://github.com/numericalalgorithmsgroup/AzureML_Best_Practice/tree/master/maskrcnn

```bash
$ git clone https://github.com/numericalalgorithmsgroup/AzureML_Best_Practice.git
$ cd maskrcnn
```

Before running any of the provided scripts you should edit `sharedconfig.py` and provide the
required information about the workspace and datasets configured in the previous section. This
configuration file is then used by the submission scripts to connect to the correct workspace and
submit jobs with the correct datasets.

We have provided two example submission and training scripts to demonstrate different potential
usage scenarios. The first script ([blobmount_submit_maskrcnn.py](blobmount_submit_maskrcnn.py))
performs training on datasets mounted over the network, the second
([localdownload_submit_maskrcnn.py](localdownload_submit_maskrcnn.py)) explicitly downloads the
dataset to the individual compute nodes before training.  These scripts are designed to run Mask
R-CNN in a benchmarking configuration. By default, the training will run for 1000 iterations with a
variable batch size of 8 images per GPU.  Optimal choices of other hyperparameters such as the
learning rate and, warmup and decay schedules will vary depending on the batch size and number of
GPUs and should be chosen accordingly.

To run either of these training scripts, first ensure you have installed the AzureML SDK as
described at the top of this tutorial, then call the script specifying the desired number of nodes
to train on. You can also select other options such as setting the number of iterations to train
over (use `--help` to see all options). For example, to benchmark for 2000 iterations on 4 nodes:

```
$ ./blobmount_submit_maskrcnn 4 --iter 2000
```

The script will then configure a 4 cluster node in the AzureML environment and submit a 2000
iteration training job to it. The cluster is configured to automatically shut down idle nodes after
60 seconds to prevent incurring unnecessary costs after the job is completed. 


## Understanding the Job Submission Scripts

In this final section of the tutorial, we will look at the steps that the submission script goes
through to create a compute cluster, configure a training environment and submit a training job.
The provided scripts automate these steps as follows:

### 1. Define Compute Environment

The `Environment` object collects configuration information about the desired state of the compute
environment such as which Python packages should be installed or what Docker container to use.
`Environment` objects are saved and versioned by the workspace to allow reuse and archive of
all information required to reproduce any given training or inference job.

To define a custom environment via the AzureML Python SDK, we must create an `Environment` object,
configure it as required and register it to the workspace. For example, to create an environment
based on a Docker image defined in a local Dockerfile (named `./MyDockerfile`)

```python
from azureml.core import Workspace, Environment

workspace = Workspace.get("AzureMLDemo")

environment = Environment("CustomDockerEnvironment")

environment.docker.enabled = True # Tell AzureML we want to use Docker
environment.docker.base_dockerfile = "./MyDockerfile" # Path to local Dockerfile
environment.python.user_managed_dependencies = True  # AzureML shouldn't try to install things

environment = environment.register(workspace) # Validate and register the environment
```

This Environment object can now be passed to the configuration of a Run and AzureML will execute
that run in the provided Docker container.

Passing a Dockerfile in this way will cause AzureML to build the Docker image in its attached Azure
Container Registry instance, however, this is typically much slower than building the container on
a dedicated host.  AzureML also supports the use of prebuilt Docker images in both public and
private container registries. The AzureML documentation on [using custom
containers](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-docker-image)
shows how to use custom docker images hosted in another Azure Container registry or in a public
registry.

#### Custom Docker Image

If you want to use a custom Docker image with AzureML it must provide the runtime dependencies
needed by AzureML. Microsoft provides a repository of [example Docker containers for
AzureML](https://github.com/Azure/AzureML-Containers) on GitHub.  The custom Docker image used for
this tutorial is based on the NVIDIA NGC container image, with the following additions:

```Dockerfile
# Build image on top of NVidia MXnet image
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${FROM_IMAGE_NAME}

# Pin Key Package Versions
ENV MOFED_VER 5.0-2.1.8.0
ENV AZML_SDK_VER 1.25.0

# Other required variables for MOFED drivers
ENV OS_VER ubuntu20.04
ENV PLATFORM x86_64

### Install Mellanox Drivers ###
RUN apt-get update && apt-get install -y libcap2 libfuse-dev && \
    wget --quiet http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update --all --without-neohost-backend --force


### Install Python Dependencies ###
RUN pip install azureml-defaults==${AZML_SDK_VER}

### Custom additions for specific training ###

# !!!! INSERT YOUR REQUIRED PACKAGE INSTALLATIONS HERE !!!!
```
This provides all the core dependencies needed to run distributed training on AzureML with PyTorch.
The NVIDIA Mask R-CNN reference implementation we are using for this tutorial is pre-installed into
the base NGC container we are using. You should add your workload and any additional dependencies
as appropriate. The complete [Dockerfile](Dockerfile) is provided in this repository along with all
the other tutorial code.

### Define cluster

Finally, we must define our desired compute resources.  AzureML supports both single compute
instances and clusters, however, as this is a *distributed* Machine Learning tutorial we will only
consider compute clusters.  Compute clusters are persistent objects and support autoscaling up to a
maximum number of nodes as well as down to zero nodes when the cluster is idle.

To create a compute cluster instance via the AzureML SDK we must first define a provisioning
configuration for the compute cluster. Then we can create the cluster with this configuration:

```python
from azureml.core import Workspace
from azureml.core.compute import AmlCompute

workspace = Workspace.get("AzureMLDemo")

cluster_config = AmlCompute.provisioning_configuration(
  vm_size="Standard_ND40rs_v2",
  min_nodes=0,
  max_nodes=16,
  idle_seconds_before_scaledown=300,
  admin_username="clusteradmin",
  admin_user_ssh_key=sshpubkey, # Contents of a public key file
  remote_login_port_public_access="Enabled"
  )

cluster = AmlCompute.create(workspace, "MyCluster", cluster_config)

cluster.wait_for_completion()
```
This will create a cluster that can scale up to 16 ND40rs v2 nodes.  The cluster will autoscale as
needed to run jobs, and idle nodes will be shut down after 5 minutes.  The last three options to
the cluster configuration allow us to specify an admin username and ssh key and enable remote ssh
access to the cluster. This can be useful for debugging or advanced usage that requires modifying
the cluster manually.

## Run training

Once we have defined datasets, a compute environment and a compute cluster, we can submit a
training job.  AzureML organises training jobs into groups or `Experiments`, each of which can then
contain many runs.

For an existing model implementation such as Mask R-CNN, there will typically already be a
training script. In this case, we can instruct AzureML to upload the script and any other
supporting config files to the compute nodes and then run the script.  This can only be done with
the AzureML SDK by creating a `ScriptRunConfig` object.  This config is then submitted to an
experiment to begin a run.

The directory `train` in the submission script repository contains several example scripts for
training Mask R-CNN.  These have been adapted from the original training scripts from the NVIDIA
repository to make them compatible with Azure.

Additionally, for distributed training on multiple nodes, we must provide information about this
distributed configuration. For this, we provide an `MpiConfiguration` object as an argument to the
`ScriptRunConfig`.

```python
from azureml.core import Workspace, Experiment, ScriptRunConfig, Dataset
from azureml.core.runconfig import MpiConfiguration
from azureml.core.compute import AmlCompute

workspace = Workspace.get("AzureMLDemo") # Get existing workspace
environment = Environment.get(workspace, "CustomDockerEnvironment") # Get existing environment
cluster = AmlCompute(workspace, "MyCluster") # Get existing cluster
dataset = Dataset.get_by_name(workspace, "coco2017_trainval")

experiment = Experiment(workspace, "MyExperiment")

dist_config = MpiConfiguration(node_count=4, process_count_per_node=8)

jobconfig = ScriptRunConfig(
  source_directory="train",
  script="./train_net.py",
  arguments=["--dataset", dataset.as_mount()],
  compute_target=cluster,
  environment=environment,
  distributed_job_config=dist_config
  )

experiment.submit(jobconfig)
```
This will upload the contents of the directory named `train`<sup>[1](#CWD_note)</sup>  and run a
training script named `train_net.py` on 4 nodes of the compute cluster with 8 processes per node (1
per GPU).  We can instruct AzureML to mount the dataset on the Compute Cluster using the
`as_mount()` method which returns the mounted location on the compute. This will be passed as an
argument to the training script so that it can locate the training data.

<a name="CWD_note"><sup>1</sup></a> The submission script should be run from the maskrcnn directory so that
the `train` directory can be found and uploaded.

### Save result data

There are multiple options for saving output data at the end of a run. The simplest is to place any
outputs into a folder named `outputs` in the working directory of the training script.  AzureML
will automatically capture all files in the `outputs` folder and store them along with all the
logging data from the run. These saved files can be downloaded later from the Machine Learning
Studio run history blade. For example to capture final weights you could add code to your training
script that saves the weights to a file `./outputs/final_weights.pkl`. Then when the run is
complete the file `final_weights.pkl` will be available, along with all the other logs and outputs,
via the AzureML Studio. More details can be found in the [AzureML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-save-write-experiment-files).

Alternatively, you can include additional code to upload results files to a separate results dataset
or directly to Azure storage.

## Next steps

If you have got to this point then you have successfully run a distributed training job,
congratulations!  Now you will most likely want to validate your new model. You can download your
model weights and perform validation locally, or you can perform the validation using AzureML.

We were excited to test our new model straight away so we downloaded it and used inferencing demo 
scripts provided to segment a bike race photo of our own:

![Image Credit: Gordon Hatton](./outputs.png)
[Image Credit: Gordon Hatton](./outputs.png)

Once you are happy with the model it can be deployed for production inferencing.  AzureML provides
a range of [MLOps features] for managing model deployment including pipeline and lifecycle
management, deployment for batch, real-time and edge inferencing scenarios. 

[MLOps features]: https://docs.microsoft.com/en-gb/azure/machine-learning/concept-model-management-and-deployment

## Summary

The Azure Machine Learning service allows fast deployment of ML workflows to the Azure cloud with
support for large file-based datasets and distributed training at scale. This tutorial provides a
complete demonstration of all the steps required to port the training of an existing Machine
Learning Workflow (Mask R-CNN) to AzureML along with a large file-based dataset.  Demonstration
submission scripts for training runs are included in the [accompanying Github repository].

[accompanying Github repository]: https://github.com/numericalalgorithmsgroup/AzureML_Best_Practice/tree/master/maskrcnn

## About NAG

[NAG](www.nag.com) has played a leading role in numerical, scientific and High Performance 
Computing (HPC) for over 50 years and is one of the few organizations that have genuine expertise
and experience in all aspects of HPC and cloud.  We offer a range of HPC and cloud services
including:

* [Cloud HPC Migration](https://www.nag.com/content/nag-cloud-hpc-migration-service)
* [Software Optimization](https://www.nag.com/content/software-modernization-service)
* [Accelerator Porting and Tuning](https://www.nag.com/content/gpus-and-accelerator-code-tuning)

as well as fully bespoke consultancy in all aspects of HPC both on-premises and in the cloud. For
more information please reach out to us at [info@nag.com](mailto:info@nag.com).

