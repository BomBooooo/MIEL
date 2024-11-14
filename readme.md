# Project Setup Guide

This guide will walk you through the setup and installation process required to configure this project.

## Prerequisites

Ensure you have **Python 3.8+** installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Installation Instructions

1. **Clone the Repository:** If you haven't already, clone this repository to your local machine:

   ```
   git clone https://github.com/BomBooooo/MIEL.git
   cd MIEL
   ```

2. **Set Up a Virtual Environment (Recommended):** It's recommended to create a virtual environment to manage dependencies:

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:** Use the provided `requirements.txt` file to install all necessary packages:

   ```
   pip install -r requirements.txt
   ```

4. **Verify Installation:** Ensure all packages are correctly installed by running:

   ```
   pip list
   ```

## Running the Project

After the dependencies are installed, you should be able to run the project as intended. Refer to the project documentation or specific scripts for further instructions on usage.

   ```
   ./scripts/MILE_ETTh1.sh
   ```

## Additional Notes

- If new dependencies are added in the future, remember to update `requirements.txt`

  ```
  pip freeze > requirements.txt
  ```

- If you encounter any issues, ensure that your Python version matches the project requirements, and that all packages are correctly installed.