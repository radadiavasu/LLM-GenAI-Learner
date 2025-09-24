# Resume Analyzer Agent
# Requirements: pip install streamlit langchain openai PyPDF2 python-docx pandas plotly

import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from typing import Optional, List, Any
import PyPDF2
import docx
import io
import json
import re
from dotenv import load_dotenv

token = load_dotenv("HF_TOKEN")


class HuggingFaceLLM(LLM):
    client: Any = None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct:nebius"
    # model_name: str = "meta-llama/Llama-3.2-3B-Instruct:novita" #- Old
    
    def __init__(self, hf_token: str):
        super().__init__()
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

class ResumeAnalyzer:
    def __init__(self, hf_token: str):
        self.llm = HuggingFaceLLM(hf_token)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text based on file type"""
        if uploaded_file.type == "application/pdf":
            return self.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.extract_text_from_docx(uploaded_file)
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return ""
    
    def create_scoring_prompt(self, resume_text: str, criteria: dict) -> str:
        """Create a detailed prompt for resume scoring"""
        prompt = f"""
        As an experienced hiring manager, analyze the following resume and provide a detailed scoring based on the specified criteria.
        
        RESUME CONTENT:
        {resume_text}
        
        SCORING CRITERIA:
        
        1. EXPERIENCE (Weight: 30%):
        Required: {criteria.get('experience', 'Not specified')}
        
        2. SKILLS (Weight: 30%):
        Required: {criteria.get('skills', 'Not specified')}
        
        3. PROJECTS (Weight: 25%):
        Required: {criteria.get('projects', 'Not specified')}
        
        4. EDUCATION (Weight: 15%):
        Required: {criteria.get('education', 'Not specified')}
        
        INSTRUCTIONS:
        - Score each category from 0-100 based on how well the resume matches the requirements
        - Provide specific reasons for your scoring
        - Consider relevance, depth, and quality of information
        - Be objective and consistent in your evaluation
        
        REQUIRED OUTPUT FORMAT (JSON):
        {{
            "overall_score": [0-100],
            "experience_score": [0-100],
            "skills_score": [0-100],
            "projects_score": [0-100],
            "education_score": [0-100],
            "experience_feedback": "detailed feedback on experience matching",
            "skills_feedback": "detailed feedback on skills matching",
            "projects_feedback": "detailed feedback on projects matching",
            "education_feedback": "detailed feedback on education matching",
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "recommendation": "overall recommendation (Strong Hire/Hire/Maybe/No Hire)"
        }}
        
        Respond ONLY with the JSON object, no additional text.
        """
        return prompt
    
    def parse_llm_response(self, response: str) -> dict:
        """Parse and clean LLM response to extract JSON"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # If no JSON found, create a default response
                return {
                    "overall_score": 0,
                    "experience_score": 0,
                    "skills_score": 0,
                    "projects_score": 0,
                    "education_score": 0,
                    "experience_feedback": "Unable to parse response",
                    "skills_feedback": "Unable to parse response",
                    "projects_feedback": "Unable to parse response",
                    "education_feedback": "Unable to parse response",
                    "strengths": ["Unable to analyze"],
                    "weaknesses": ["Unable to analyze"],
                    "recommendation": "Unable to analyze"
                }
        except json.JSONDecodeError:
            st.error("Error parsing LLM response as JSON")
            return {
                "overall_score": 0,
                "experience_score": 0,
                "skills_score": 0,
                "projects_score": 0,
                "education_score": 0,
                "experience_feedback": "Error in analysis",
                "skills_feedback": "Error in analysis",
                "projects_feedback": "Error in analysis",
                "education_feedback": "Error in analysis",
                "strengths": ["Error in analysis"],
                "weaknesses": ["Error in analysis"],
                "recommendation": "Error in analysis"
            }
    
    def analyze_resume(self, resume_text: str, criteria: dict, filename: str) -> dict:
        """Analyze a single resume"""
        prompt = self.create_scoring_prompt(resume_text, criteria)
        
        try:
            with st.spinner(f"Analyzing {filename}..."):
                response = self.llm._call(prompt)
                result = self.parse_llm_response(response)
                result['filename'] = filename
                return result
        except Exception as e:
            st.error(f"Error analyzing {filename}: {str(e)}")
            return {
                'filename': filename,
                'overall_score': 0,
                'experience_score': 0,
                'skills_score': 0,
                'projects_score': 0,
                'education_score': 0,
                'experience_feedback': f"Error: {str(e)}",
                'skills_feedback': f"Error: {str(e)}",
                'projects_feedback': f"Error: {str(e)}",
                'education_feedback': f"Error: {str(e)}",
                'strengths': [f"Error: {str(e)}"],
                'weaknesses': [f"Error: {str(e)}"],
                'recommendation': "Error in analysis"
            }

def create_visualizations(results: List[dict]):
    """Create visualizations for the analysis results"""
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Overall scores bar chart
    fig_overall = px.bar(
        df, 
        x='filename', 
        y='overall_score',
        title='Overall Resume Scores',
        color='overall_score',
        color_continuous_scale='RdYlGn',
        text='overall_score'
    )
    fig_overall.update_traces(texttemplate='%{text}', textposition='outside')
    fig_overall.update_layout(xaxis_tickangle=-45)
    
    # Category-wise scores radar chart
    categories = ['experience_score', 'skills_score', 'projects_score', 'education_score']
    fig_radar = go.Figure()
    
    for idx, result in enumerate(results):
        fig_radar.add_trace(go.Scatterpolar(
            r=[result[cat] for cat in categories],
            theta=['Experience', 'Skills', 'Projects', 'Education'],
            fill='toself',
            name=result['filename'][:15] + '...' if len(result['filename']) > 15 else result['filename']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Category-wise Score Comparison"
    )
    
    # Category scores grouped bar chart
    df_melted = df.melt(
        id_vars=['filename'], 
        value_vars=categories,
        var_name='Category', 
        value_name='Score'
    )
    df_melted['Category'] = df_melted['Category'].str.replace('_score', '').str.title()
    
    fig_grouped = px.bar(
        df_melted, 
        x='filename', 
        y='Score',
        color='Category',
        barmode='group',
        title='Detailed Category Scores',
        text='Score'
    )
    fig_grouped.update_traces(texttemplate='%{text}', textposition='outside')
    fig_grouped.update_layout(xaxis_tickangle=-45)
    
    return fig_overall, fig_radar, fig_grouped

def main():
    st.set_page_config(
        page_title="Resume Analyzer Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Resume Analyzer Agent")
    st.markdown("Upload multiple resumes and get detailed AI-powered analysis based on your hiring criteria.")
    
   
    hf_token = token
        
    if not hf_token:
        st.warning("Please enter your Hugging Face token to continue.")
        st.stop()
    
    # Initialize analyzer
    try:
        analyzer = ResumeAnalyzer(hf_token)
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload 2 or more resume files for analysis"
        )
        
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 resumes for comparison.")
        else:
            st.success(f"✅ {len(uploaded_files)} resumes uploaded")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.type})")
    
    with col2:
        st.header("🎯 Hiring Criteria")
        st.markdown("Define what you're looking for in the ideal candidate:")
        
        experience_req = st.text_area(
            "Experience Requirements",
            placeholder="e.g., 3+ years in Python development, experience with web frameworks, API development...",
            help="Describe the required experience level and background"
        )
        
        skills_req = st.text_area(
            "Required Skills",
            placeholder="e.g., Python, React, SQL, Machine Learning, Communication skills...",
            help="List the technical and soft skills required"
        )
        
        projects_req = st.text_area(
            "Project Requirements",
            placeholder="e.g., Full-stack web applications, ML models, open-source contributions...",
            help="Describe the types of projects that demonstrate relevant experience"
        )
        
        education_req = st.text_area(
            "Education Requirements",
            placeholder="e.g., Bachelor's in Computer Science or related field, relevant certifications...",
            help="Specify educational background requirements"
        )
    
    # Analysis section
    if len(uploaded_files) >= 2 and all([experience_req, skills_req, projects_req, education_req]):
        
        if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
            
            criteria = {
                'experience': experience_req,
                'skills': skills_req,
                'projects': projects_req,
                'education': education_req
            }
            
            # Extract text from all resumes
            resume_texts = {}
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    text = analyzer.extract_text_from_file(file)
                    if text:
                        resume_texts[file.name] = text
                    progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if not resume_texts:
                st.error("No text could be extracted from the uploaded files.")
                st.stop()
            
            # Analyze all resumes
            st.header("🔍 Analysis Results")
            results = []
            
            progress_bar = st.progress(0)
            for idx, (filename, text) in enumerate(resume_texts.items()):
                result = analyzer.analyze_resume(text, criteria, filename)
                results.append(result)
                progress_bar.progress((idx + 1) / len(resume_texts))
            
            if results:
                # Sort results by overall score
                results.sort(key=lambda x: x['overall_score'], reverse=True)
                
                # Display summary table
                st.subheader("📊 Summary Rankings")
                summary_df = pd.DataFrame([{
                    'Rank': idx + 1,
                    'Resume': result['filename'],
                    'Overall Score': f"{result['overall_score']}/100",
                    'Experience': f"{result['experience_score']}/100",
                    'Skills': f"{result['skills_score']}/100",
                    'Projects': f"{result['projects_score']}/100",
                    'Education': f"{result['education_score']}/100",
                    'Recommendation': result['recommendation']
                } for idx, result in enumerate(results)])
                
                st.dataframe(summary_df, use_container_width=True)
                
                # # Visualizations
                # st.subheader("📈 Visual Analysis")
                # fig_overall, fig_radar, fig_grouped = create_visualizations(results)
                
                # tab1, tab2, tab3 = st.tabs(["Overall Scores", "Category Comparison", "Detailed Breakdown"])
                
                # with tab1:
                #     st.plotly_chart(fig_overall, use_container_width=True)
                
                # with tab2:
                #     st.plotly_chart(fig_radar, use_container_width=True)
                
                # with tab3:
                #     st.plotly_chart(fig_grouped, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Analysis")
                
                for idx, result in enumerate(results):
                    with st.expander(f"#{idx+1} {result['filename']} - Score: {result['overall_score']}/100"):
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Category Scores:**")
                            st.write(f"• Experience: {result['experience_score']}/100")
                            st.write(f"• Skills: {result['skills_score']}/100")
                            st.write(f"• Projects: {result['projects_score']}/100")
                            st.write(f"• Education: {result['education_score']}/100")
                            
                            st.write(f"**Recommendation:** {result['recommendation']}")
                        
                        with col2:
                            st.write("**Strengths:**")
                            for strength in result['strengths']:
                                st.write(f"✅ {strength}")
                            
                            st.write("**Areas for Improvement:**")
                            for weakness in result['weaknesses']:
                                st.write(f"⚠️ {weakness}")
                        
                        st.write("**Detailed Feedback:**")
                        
                        feedback_tabs = st.tabs(["Experience", "Skills", "Projects", "Education"])
                        
                        with feedback_tabs[0]:
                            st.write(result['experience_feedback'])
                        
                        with feedback_tabs[1]:
                            st.write(result['skills_feedback'])
                        
                        with feedback_tabs[2]:
                            st.write(result['projects_feedback'])
                        
                        with feedback_tabs[3]:
                            st.write(result['education_feedback'])
                
                # Download results
                st.subheader("💾 Export Results")
                results_df = pd.DataFrame(results)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Detailed Results as CSV",
                    data=csv,
                    file_name=f"resume_analysis_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()