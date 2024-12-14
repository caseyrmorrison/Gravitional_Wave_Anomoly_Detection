### Project Overview: 
Based on the NSF HDR A3D3: Detecting Anomalous Gragvitational Wave Signals description this is what we are tackling:

The target is to identify anomalous gravity wave sources with a deep learning algorithm. This task is a anomaly detection task.

Gravitational waves are ripples in spacetime caused by cosmic events, predicted by Einstein in 1915 and detected using LIGO, which has two laser interferometers in Washington and Louisiana. The dual setup verifies signals, reduces noise, and enhances sensitivity, enabling accurate detection of gravitational wave sources.

The Anomaly Detection Challenge focuses on analyzing LIGO's O3a observation data, comprising cleaned time series with known gravitational wave (GW) events removed. The dataset includes real noise background data and simulated signals injected into this noise. Participants must develop models to detect anomalies, leveraging background data and simulated signals to improve detection. This challenge bridges astrophysics and machine learning, driving innovation in detecting gravitational waves.

**Goal**: We want high reproduction errors on the output and that means an anomaly exists

### Google Colab - Option
The project was ran in one Google Colab and has been modified to fit this format. All training was done on Colab with an A100 otherwise it will take too long locally. This code uses a much smaller epoch than used on Colab to show functionality over results. 
How to Colab:
1. Either upload the datasets to top level colab folder or use the Google Drive code edited for your path of the datasets.
2. Need to be using a GPU
3. Run all
4. All output will be either saved to the top level or output to the screen

### Local Setup Instructions:
Step-by-step instructions to set up the environment, including how to install dependencies from requirements.txt. This is written for MacOS and Linux. Use windows commands or powershell in place of Linux commands.
1. Open a terminal and navigate to the top level of the project 
ex. `cd EEP596_Project_CaseyMorrison`
2. Run setup script
    a. Run: `./setup.sh`
    b. If not executable run: `chmod +x setup.sh`
    c. This will setup a python virtual environment and install from requirements.txt

### How to Run:
1. If no terminal is open, open one and navigate to src directory: `cd src`
    a. This code is setup for 2 epochs for local execution as it will take a while for training, adjust inside main.py if you want to use more.
2. Run: `python main.py`
3. Monitor output
4. When completed output will go to the results folder in the form of png files and the model will be saved under checkpoints

### Expected Output: 
- Output is expected to be the loss, the ROC curve to see the AUC score to see the performance of the training. 
- The curve wants to be above the diagonal on the ROC curve. 
- A histogram is also output to visualize all the recreation errors distribution. 
- Model Architecture is output to the console
- If you navigate to demo and run demo.py you can see the output for reconstruction errors used for submitting to the hackathon.
- `cd demo`
- `python demo.py` 

### Pre-trained Model Link: 
A pretrained model with good performance
https://drive.google.com/file/d/1-2fVvMrn1NXXdwSZq2-14JEcDtEholjZ/view?usp=share_link

### Acknowledgments: 
Dataset comes from the NSF HDR A3D3 codabench: https://www.codabench.org/competitions/2626/
The data was captured by LIGO's O3a observation period.

### Hyperparameters and training setup
I used many different hyperparameters and training variations during testing but some of my best results used these specification:
- head_size=64,
- num_heads=2,
- ff_dim=64,
- num_transformer_blocks=4,
- num_dense_blocks=2,
- dropout=0.2,
- l2_reg=1e-6
- Variable training rate based on performance starting at 1e-4 and ending at 1e-6
- Used l2 regularization
- Used an A100 for quicker training otherwise even just 50 epochs could take hours.
- epochs=100
- batch_size=64
- used early stopping as well