import gradio as gr
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load saved data
def load_saved_data():
    try:
        with open('job_descriptions.pkl', 'rb') as f:
            job_descriptions = pickle.load(f)
        job_embeddings = np.load('job_embeddings.npy')
        print(f"Loaded {len(job_descriptions)} job descriptions")
        return job_descriptions, job_embeddings
    except Exception as e:
        print(f"Error loading data: {e}")
        return [], None

def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()

# Initialize model and load data
model = SentenceTransformer('all-MiniLM-L6-v2')
job_descriptions, job_embeddings = load_saved_data()

def format_fit_label(label: str) -> str:
    """Format the fit label with color and emoji."""
    if label == "Good Fit":
        return "🟢 Good Fit"
    elif label == "Potential Fit":
        return "🟡 Potential Fit"
    else:
        return "🔴 No Fit"

def search_jobs(query: str, num_results: int, min_experience: int = 0, 
                location_preference: str = "", work_mode_preference: str = "",
                min_fit_level: str = "Any") -> str:
    """
    Search for jobs based on query and filters.
    """
    try:
        # Encode query
        processed_query = preprocess_text(query)
        query_embedding = model.encode([processed_query])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], job_embeddings)[0]
        
        # Apply filters and sort
        filtered_results = []
        for idx, similarity in enumerate(similarities):
            job = job_descriptions[idx].copy()
            
            # Apply filters
            if location_preference and location_preference.lower() not in job['location'].lower():
                continue
            if work_mode_preference != "Any" and work_mode_preference.lower() not in job['work_mode'].lower():
                continue
            if min_fit_level != "Any":
                if min_fit_level == "Good Fit" and job['fit_label'] != "Good Fit":
                    continue
                if min_fit_level == "Potential Fit" and job['fit_label'] == "No Fit":
                    continue
            
            job['similarity_score'] = float(similarity)
            filtered_results.append(job)
        
        # Sort by similarity score
        filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Get top k results
        results = filtered_results[:num_results]
        
        # Format output
        if not results:
            return "No matching jobs found. Try adjusting your search criteria."
        
        formatted_output = f"Found {len(results)} matching jobs:\n"
        for i, job in enumerate(results, 1):
            formatted_output += f"\n{i}. Role: {job['role']}\n"
            formatted_output += f"   Location: {job['location']}\n"
            formatted_output += f"   Work Mode: {job['work_mode']}\n"
            formatted_output += f"   Match Score: {job['similarity_score']:.2f}\n"
            formatted_output += f"   Fit Assessment: {format_fit_label(job['fit_label'])}\n"
            formatted_output += "\n   Job Description:\n"
            desc_lines = job['text'].split('\n')
            formatted_output += "\n".join(f"   {line}" for line in desc_lines) + "\n"
            formatted_output += "-" * 80 + "\n"
        
        return formatted_output
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=search_jobs,
    inputs=[
        gr.Textbox(
            lines=2,
            placeholder="Enter your skills, experience, and preferences...",
            label="Job Search Query"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Number of results"
        ),
        gr.Number(
            value=0,
            label="Minimum Years of Experience",
            minimum=0,
            maximum=20
        ),
        gr.Textbox(
            lines=1,
            placeholder="Enter preferred location (optional)",
            label="Location Preference"
        ),
        gr.Radio(
            choices=["Any", "Remote", "Hybrid", "Onsite"],
            value="Any",
            label="Work Mode Preference"
        ),
        gr.Radio(
            choices=["Any", "Potential Fit", "Good Fit"],
            value="Any",
            label="Minimum Fit Level"
        )
    ],
    outputs=gr.Textbox(
        lines=25,
        label="Search Results",
        show_copy_button=True
    ),
    title="Job Search Assistant",
    description="Search for relevant job positions based on your skills and preferences.",
    examples=[
        ["I am skilled in Python and SQL with 5 years of experience in data analytics", 5, 3, "New York", "Any", "Any"],
        ["Experienced Business Analyst with banking domain knowledge", 3, 5, "", "Remote", "Potential Fit"],
        ["Entry level software developer with Java experience", 5, 0, "", "Hybrid", "Any"]
    ]
)

if __name__ == "__main__":
    iface.launch()