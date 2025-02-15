## Quick Start

Follow these steps to get started quickly:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/sunshine-JLU/deepseek-janus-pro-lora.git

   cd deepseek-janus-pro-lora

   
2. **Enviroment**  
   ```bash
   conda create -n janus-pro-lora python=3.10 -y

   conda init

   source ~/.bashrc
   
   conda activate janus-pro-lora
   
   pip install -r requirements.txt

3. **Download the Model**  
   Download the Janus-Pro-7B model:
   ```bash
   modelscope download --model deepseek-ai/Janus-Pro-7B --local_dir ./Janus-Pro-7B

4. **Prepare the Dataset**  
   Prepare your dataset or use my demo dataset to train model:
   ```bash
   unzip trump.zip

   python process_image_with_description.py

5. **Run the Notebook**
   ```bash
   python -m ipykernel install --user --name janus-pro-lora --display-name "Python (janus-pro-lora)"
   
  Open and run the deepseek-janus-pro-lora.ipynb notebook to start fine-tuning the model.
  Open and run the janus-pro-lora-inference.ipynb notebook to evaluate the model.

## Hardware requirement

GPU Memory at least 32GB would not appear OOM problem.
![janus](https://github.com/user-attachments/assets/e3d91ada-5a6e-402e-b9fc-2699955abd75)


 

