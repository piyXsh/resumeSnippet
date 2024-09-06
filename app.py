import streamlit as st
from PyPDF2 import PdfReader
import re
import pickle
import os

# Load models and vectorizers
resume_categorizer = pickle.load(open("Resume_Categorization/model.pkl", "rb"))
tfidf_categorizer = pickle.load(open("Resume_Categorization/tfidf_categorizer.pkl", "rb"))
job_recommender = pickle.load(open("Job_recommendation/model1.pkl", "rb"))
tfidf_recommender = pickle.load(open("Job_recommendation/tfidf_recommender.pkl", "rb"))

label_encoder = pickle.load(open("Resume_Categorization\label_encoder.pkl", 'rb'))

# Function to convert PDF to text
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Function to clean resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)  # remove URL
    clean_text = re.sub('RT|cc', ' ', clean_text)  # remove RT and cc
    clean_text = re.sub('#\S+\s', ' ', clean_text)  # remove hashtags
    clean_text = re.sub('@\S+', ' ', clean_text)  # remove mentions
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)  # remove punctuations
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)  # remove non-ASCII characters
    clean_text = re.sub('\s+', ' ', clean_text)  # remove whitespaces
    return clean_text

# Function to predict category
def predict_category(resume_text):
    resume_text = clean_resume(resume_text)
    resume_tfidf = tfidf_categorizer.transform([resume_text])
    predicted_label = resume_categorizer.predict(resume_tfidf)[0]
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_category

# Function to predict job
def predict_job(resume_text):
    resume_text = clean_resume(resume_text)
    resume_tfidf = tfidf_recommender.transform([resume_text])
    recommended_job = job_recommender.predict(resume_tfidf)[0]
    return recommended_job

# Function to extract contact number
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+\(91\)\s?)?\d{5}[-\s]?\d{5}\b|\b(?:\+91|91|0)?\d{10}\b|(?:\+91-?|91-?)\d{10}|\(\d{3}\)\s?\d{8}|\d{3}-\d{8}\b"
    matches = re.findall(pattern, text)
    return ", ".join(matches) if matches else "No valid contact number found"
   
# Function to extract urls from resume
def extract_links_from_resume(text):
    pattern = r'(https?://[^\s]+)'
    matches = re.findall(pattern, text)
    return ", ".join(matches) if matches else "No valid links found"


# Function to extract email
def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email

# Function to extract skills
def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib', 'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition', 'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks', 'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees', 'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN', 'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL', 'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker', 'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption', 'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration', 'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development', 'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite', 'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research', 'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing', 'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)', 'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting', 'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing', 'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing', 'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation', 'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento', 'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement', 'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL', 'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics', 'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL', 'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL', 'Socket.io', 'WebSockets',
        'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#', 'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development', 'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI', 'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda', 'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration', 'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch', 'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment', 'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation', 'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling', 'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya', '3ds Max', 'Cinema 4D', 'Substance Painter',
        'Character Rigging', 'Environment Design', 'Texture Painting', 'Shader Programming', 'Physics Simulation', 'AI in Games', 'Procedural Generation'
    ]
    skills = [skill for skill in skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    return skills

# Function to extract education
def extract_education_from_resume(text):
    education_keywords = [
        'Bachelor', 'Master', 'PhD', 'Doctorate', 'Diploma', 'Certificate', 'Associate', 'Undergraduate', 'Graduate', 'Postgraduate',
        'Engineering', 'Computer Science', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Economics', 'Business Administration',
        'Management', 'Finance', 'Accounting', 'Marketing', 'Statistics', 'Data Science', 'Information Technology', 'Cybersecurity',
        'Artificial Intelligence', 'Machine Learning', 'Deep Learning', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering',
        'Civil Engineering', 'Environmental Science', 'Geology', 'Political Science', 'Sociology', 'Psychology', 'Linguistics', 'Philosophy',
        'History', 'International Relations', 'Law', 'Human Resources', 'Public Administration', 'Education', 'Healthcare', 'Medicine',
        'Nursing', 'Pharmacy', 'Public Health', 'Social Work', 'Arts', 'Design', 'Music', 'Performing Arts'
    ]
    education = [keyword for keyword in education_keywords if re.search(rf"(?i)\b{re.escape(keyword)}\b", text)]
    return education

# Function to extract name
def extract_name_from_resume(text):
    name = None
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()
    return name

def main():
    st.title("Resume Snippet")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = pdf_to_text(uploaded_file)
        elif uploaded_file.type == "text/plain":
            resume_text = str(uploaded_file.read(), 'utf-8')
        else:
            st.write("Invalid file format")
            return
        
        cleaned_text= clean_resume(resume_text)

        # Display extracted information
        st.subheader("Extracted Information")

        st.write("**Category Prediction:**")
        predicted_category = predict_category(cleaned_text)
        st.write(predicted_category)

        st.write("**Job Recommendation:**")
        recommended_job = predict_job(cleaned_text)
        st.write(recommended_job)

        st.write("**Contact Number:**")
        contact_number = extract_contact_number_from_resume(cleaned_text)
        st.write(contact_number)

        st.write("**Email:**")
        email = extract_email_from_resume(resume_text)
        st.write(email)

        st.write("**Skills:**")
        skills = extract_skills_from_resume(cleaned_text)
        st.markdown(f'<ul>{"".join(f"<li>{skill}</li>" for skill in skills)}</ul>', unsafe_allow_html=True)

        st.write("**Education:**")
        education = extract_education_from_resume(cleaned_text)
        st.markdown(f'<ul>{"".join(f"<li>{edu}</li>" for edu in education)}</ul>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()