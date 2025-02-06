## Quick Start

Follow these steps to get started quickly:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/sunshine-JLU/deepseek-r1-distill-llama-8b-lora.git

   
2. **Enviroment**  
   ```bash
   pip install -r requirements.txt

3. **Download the Model**  
  Set the Hugging Face endpoint and download the deepseek-r1-distill-llama-8b model:
   ```bash
   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/deepseek-r1-distill-llama-8b --local-dir ./deepseek-r1-distill-llama-8b --resume-download --cache-dir ./cache

4. **Run the Notebook**  
  Open and run the deepseek-r1-distill-llama-8b.ipynb notebook to start fine-tuning the model.

## Hardware requirement

GPU Memory at least 48GB would not appear OOM problem.
![微信图片_20250206162602](https://github.com/user-attachments/assets/e4232a2e-4e5d-4636-921e-d9e6e4855134)

 

