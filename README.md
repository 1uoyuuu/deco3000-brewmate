# CoffeeGPT Setup Instructions

## Introduction

This document provides a step-by-step guide to setting up the necessary environment for running the provided Streamlit application which leverages various APIs for functionality.

## Prerequisites

Before starting, ensure that you have Python installed on your machine. This project is tested on Python 3.8 and above.

## Installation

### 1. Clone the repository

Clone this repository to your local machine to get started.

### 2. Install Required Packages

To install the necessary packages, navigate to the project directory and run:

`pip install -r requirements.txt`

This will install all the dependencies listed in the `requirements.txt` file.

### 3. Check OpenAI Package Version

Make sure that the `openai` package version is `0.28.1`. The newer versions may not be compatible with this project. To check the version, run:

`pip show openai`

If you have a different version installed, you can install the correct one using:

`pip install openai==0.28.1`

### 4. API Keys Configuration

Create a `.env` file in the root directory of this project and include the following keys:

`OPENAI_API_KEY='your_openai_api_key_here'`
`BROWSERLESS_API_KEY='your_browserless_api_key_here'`
`SERPER_API_KEY='your_serper_api_key_here'`

Replace the placeholders with your actual API keys.

### 5. Obtain API Keys

If you don't already have the necessary API keys, you can obtain them from the following services (they offer free tiers for limited usage):

- OpenAI: [OpenAI API](https://beta.openai.com/signup/)
- Browserless: [browserless.io](https://www.browserless.io/)
- Serper: [serper.dev](https://serper.dev/)

## Running the Application

Once you have all the configuration set, run the application using Streamlit with the following command in your terminal:
`streamlit run app.py`
Navigate to the provided localhost URL in your web browser to view the application.

## Support

For any issues or support, please file an issue on the repository's issue tracker.

Thank you for using our application!
