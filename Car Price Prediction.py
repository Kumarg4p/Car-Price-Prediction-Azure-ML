#!/usr/bin/env python
# coding: utf-8

# # Connect to Azure ML Workspace

# In[1]:


import azureml.core
from azureml.core import Workspace
ws=Workspace.from_config()
print('Workspace name: ' + ws.name,
      'Azure region: ' + ws.location,
      'Subscription id: ' + ws.subscription_id,
      'Resource group: ' + ws.resource_group, sep='\n')


# # Get training data already registerd with Azure ML

# In[2]:


import pandas as pd
from azureml.core import Dataset
cars = ws.datasets['car_data'].to_pandas_dataframe()
cars.info()


# # Create Compute Target

# In[3]:


from azureml.core.compute import AmlCompute, ComputeTarget
clust_name = 'car1clust'
try:
    compute_target = ComputeTarget(workspace=ws, name=clust_name)
    print('Found existing compute target.')
except:
    compute_config=AmlCompute.provisioning_configuration(vm_size='Standard_DS11_v2',
                                                         vm_priority='lowpriority',
                                                         idle_seconds_before_scaledown=3000,
                                                         min_nodes=0,
                                                         max_nodes=2)
    compute_target = ComputeTarget.create(ws, clust_name, compute_config)
    compute_target.wait_for_completion(show_output=True)


# # Create Scripts for pipeline steps

# In[4]:


import os
#create a folder for the pipeline step files
experiment_folder = 'carprice_pipeline'
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)


# In[5]:


get_ipython().run_cell_magic('writefile', '$experiment_folder/prep_carprice.py', '#import libraries\nimport os\nimport argparse\nimport pandas as pd\nimport numpy as np\nfrom azureml.core import Run\n\n#Get parameters\nparser = argparse.ArgumentParser()\nparser.add_argument("--input-data", type=str, dest=\'raw_data\', help=\'raw dataset\')\nparser.add_argument("--prepped-data", type=str, dest=\'prepped_data\', help=\'Output folder\')\nargs=parser.parse_args()\nsave_folder = args.prepped_data\n\n#get the experiment run context\nrun = Run.get_context()\n\n#Load the data (passed as input dataset)\nprint("Loading data...")\ncar = run.input_datasets[\'raw_data\'].to_pandas_dataframe()\n\n# log shape of DF\n#run.log_list(\'shape\', car.shape)\n\n\n#log the unique values of categorical features\n#run.log_list(\'Fuel Type\', car[\'Fuel_Type\'].unique)\n#run.log_list(\'Seller Type\', car[\'Seller_Type\'].unique)\n#run.log_list(\'Transmission\', car[\'Transmision\'].unique)\n#run.log_list(\'Owner\', car[\'Owner\'].unique)\n\n#dropping unwanted feature\ncar_finaldf = car[[\'Year\', \'Selling_Price\', \'Present_Price\', \'Kms_Driven\',\n       \'Fuel_Type\', \'Seller_Type\', \'Transmission\', \'Owner\']]\n\n#add a new feature as Current year\ncar_finaldf[\'Current_year\']=2021\n\n#calculating the age of the car\ncar_finaldf [\'number_of_years\']=car_finaldf [\'Current_year\']-car_finaldf [\'Year\']\n\n#Dropping year and Curren year\ncar_finaldf.drop([\'Year\'], axis=1, inplace=True)\ncar_finaldf.drop([\'Current_year\'], axis=1, inplace=True)\n\n#Handling categorical columns\ncar_finaldf=pd.get_dummies(car_finaldf,drop_first=True)\n\n# save the prepped data\nprint("Saving Data...")\nos.makedirs(save_folder, exist_ok=True)\nsave_path = os.path.join(save_folder, \'data.csv\')\ncar_finaldf.to_csv(save_path, index=False, header=True)\n\n#End the run\nrun.complete()')


# In[11]:


get_ipython().run_cell_magic('writefile', '$experiment_folder/train_carprice.py', '#import libraries\nfrom azureml.core import Run, Model\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport os\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn import metrics\n\n#Get parameteres\nparser = argparse.ArgumentParser()\nparser.add_argument("--training-data", type=str, dest=\'training_data\', help=\'training data\')\nargs = parser.parse_args()\ntraining_data = args.training_data\n\n# Get the experiment run context\nrun = Run.get_context()\n\n# load the prepared data file in the training folder\nprint("Loading Data...")\nfile_path = os.path.join(training_data,\'data.csv\')\ncar_finaldf = pd.read_csv(file_path)\n\n# Separate features and labels\nX=car_finaldf.iloc[:,1:]\ny=car_finaldf.iloc[:,0]\n\n# Split data into training set and test set\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)\n\n# train RandomForestRegerssor\nmodel_rf=RandomForestRegressor(n_estimators=1000, min_samples_split=2, min_samples_leaf=1, max_features=\'sqrt\',\n                         max_depth=25)\nmodel_rf.fit(X_train, y_train)\n# predict the car price\ny_pred = model_rf.predict((X_test))\n\n#Evaluate the model\nr_square = metrics.r2_score(y_test, y_pred)\nrun.log(\'R-square\',r_square)\n\n#save the trained model in the outputs folder\nprint("saving model...")\nos.makedirs(\'outputs\', exist_ok=True)\nmodel_file = os.path.join(\'outputs\', \'carprice_model.pkl\')\njoblib.dump(value=model_rf,filename=model_file)\n\n#Register the model\nprint(\'Registering the model...\')\nModel.register(workspace=run.experiment.workspace,\n               model_path = model_file,\n               model_name = \'carprice_model\',\n               tags ={\'Training context\':\'Pipeline\'},\n               properties={\'R_square\': r_square})\n\nrun.complete()')


# # Create environment file

# In[7]:


get_ipython().run_cell_magic('writefile', '$experiment_folder/experiment_env.yml', 'name: experiment_env\ndependencies:\n- python=3.6.2\n- scikit-learn\n- ipykernel\n- matplotlib\n- pandas\n- pip\n- pip:\n  - azureml-defaults\n  - pyarrow')


# In[12]:


from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration

#Create python environment 
experiment_env = Environment.from_conda_specification("experiment_env", experiment_folder + "/experiment_env.yml")

#Register the environment
experiment_env.register(workspace=ws)
registered_env = Environment.get(ws,'experiment_env')

#create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

#assign the compute target
pipeline_run_config.target = clust_name

#assign the environmet to the run configuration
pipeline_run_config.environment = registered_env

print("Run config created")


# # Create and run pipeline

# In[13]:


from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep

#Get the training dataset
car_data = ws.datasets.get("car_data")

# create an OuputFileDatasetConfig (Temporary data ref) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig("prepped_data")

#step 1, Run the data prep script

prep_step = PythonScriptStep(name = " Prepare Data",
                            source_directory = experiment_folder,
                            script_name = "prep_carprice.py",
                            arguments = ['--input-data', car_data.as_named_input('raw_data'),
                                         '--prepped-data', prepped_data],
                            compute_target = clust_name,
                            runconfig = pipeline_run_config,
                            allow_reuse = True)

#step 2, run the training script
train_step = PythonScriptStep(name = "Train and Register Model",
                             source_directory = experiment_folder,
                             script_name = 'train_carprice.py',
                             arguments = ['--training-data', prepped_data.as_input()],
                             compute_target = clust_name,
                             runconfig = pipeline_run_config,
                             allow_reuse = True)

print("Pipeline steps defined")


# In[14]:


from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'carprice-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)


# In[15]:


for run in pipeline_run.get_children():
    print(run.name, ':')
    metrics = run.get_metrics()
    for metric_name in metrics:
        print('\t',metric_name, ":", metrics[metric_name])


# In[16]:


from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


# In[17]:


# Publish the pipeline from the run
published_pipeline = pipeline_run.publish_pipeline(
    name="carprice-pipeline", description="Train car price prediction model", version="1.0")

published_pipeline


# In[18]:


rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

