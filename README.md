# Project Title
An easy-to-use fluorescent cell counter and viability calculator for blue (DAPI), green (live), and red (dead) stains featuring customizable recognition thresholds, segmentation previews, algorithm-supported threshold recommendations, live/dead percentages, and automatic scale-bar removal.

## Biomedical Context
This tool is designed for students, laboratory trainees, and researchers learning how to quantify cell viability using flrourescence microscopy.

In many cell-culture experiments, viability is assessed using multicolor stains such as:
  - DAPI for nuclei (Blue)
  - Calcein-AM or GFP for live cells (green)
  - Ethidium Homordimer/ PI (EthD-1) for dead cells (red)
Manually counting cells is time-consuming and prone to inconsistencies. 

This app provides a fast, automated, and interactive way to calculate cell counta and viability percentage tool in the biomedical engineering courses and a practical early-stage analysis tool for imaging-based experiments. 


## Quick Start Instructions

### Opening the Repository in GitHub Codespaces
1. Navigate to your GitHub repository.
2. Lick "Code" -> "Codespaces" -> "Create Codespace on main."
3. Wait for the Codespace environment to fully load. 
### Running the Application
1. Open GitHub Codespace
2. Install dependencies with `pip install -r requirements.txt` in the terminal
3. Run Streamlit with `streamlit run app.py --server.address 0.0.0.0 --server.port 8501` in the terminal
4. Open the app via Codespace UI at port 8501 or by using the URL provided in the terminal

## Usage Guide
**Step 1:** 
- Input cell stain images
- Upload one image for each stain:
    + Blue (DAPI)
    + Green (Live)
    + Red (Dead)
      
<img width="800" height="500" alt="Screen Shot 2025-12-08 at 2 27 42 PM" src="https://github.com/user-attachments/assets/b0cb2ad4-9f3a-460d-9de9-f832d143f861" />

**Step 2:** 
- Set color and minumum area thresholds for cell recognition
- Use the sidebar sliders or exact numeric inputs
- Recommendations automatically appear after image upload
  
<img width="100" height="530" alt="Screen Shot 2025-12-08 at 2 39 34 PM" src="https://github.com/user-attachments/assets/612cfe07-3bd9-4a56-b85b-22f0137f238c" />

**Step 3:** 
- Click "Run Analysis" and obtain results
- Outputs include:
    + Per-channel cell counts
    + Total nuclei, live cells, and dead cells
    + Live/Dead percentages
    + Segmentation previews (origina -> cleaned -> binary mask)
  
<img width="600" height="300" alt="Screen Shot 2025-12-08 at 2 37 18 PM" src="https://github.com/user-attachments/assets/98b3f7ee-3223-4ab3-8229-4272deedb2c7" />

<img width="600" height="300" alt="Screen Shot 2025-12-08 at 2 37 50 PM" src="https://github.com/user-attachments/assets/1cdc1c4d-4c0d-422c-8e6f-4fa7a26fa9b9" />

## Data Description 
This app processes three-channel fluorescence microscopy datasets in which each cannel is acquired and saves as a separate image file. The required inputs include: 
  - DAPI (blue): nuclei count
  - Live (green): metabolically active, membrane-intact cells
  - Dead (red): membrane-compromised cells

The images used during testing consisted of: 
- ~400-500 DAPI-labeled nuclei
- High-density green fluorescence indicaiting live cells
- Low-intesnity red fluorescence representing dead cells
  
### Data Source
The sample fluorescence images used for testing (DAPI, Live and Dead channels) were collected from previous biomedical engineering cell culture labs at the University of Florida involving mammalian cell culture and standard LIVE/DEAD viability staining (Calcein-AM and EthD-1). These images are intended exclusively for eduational use and should not be redistributed without instructor permission. 

Users may upload any of the following file fypes: **.png, .jpg, .jpeg, .tif, .tiff.**

## Project Structure
├─ app.py                      # Main Streamlit application
├─ requirements.txt            # Python dependencies
├─ README.md                   # Project documentation
├─ data/                       # Optional folder containing example images
├─ .devcontainer/              # GitHub Codespaces environment files
└─ utils/                      # Optional helper modules (if added)

Key Components
- app.py: Image upload, preprocessing, segmentation, threshold recommendations and output visualization.
- requirements.txt: Ensures reproducible environment setup.
- data/: Sample text images (if included).
- utils/: For helper functions (optional) 

