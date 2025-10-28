import streamlit as st
import nltk
import spacy
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import yt_dlp as youtube_dl
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time, datetime
import pandas as pd
import base64
from Courses import ds_videos, web_videos, android_videos, ios_videos, uiux_videos, resume_videos, interview_videos

# Ensure NLTK data is downloaded
nltk.download('stopwords')

# Ensure spaCy model is installed
try:

    nlp = spacy.load('en_core_web_trf')
except OSError:
    st.error("The 'en_core_web_trf' model is not installed. Please run `python -m spacy download en_core_web_sm` in your terminal.")
    st.stop()

# Function to load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()



# Load Lottie animation
lottie_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import language_tool_python  # For grammar and readability checks

# Function to calculate advanced ATS score
def calculate_ats_score(resume_text, job_description):
    # Keyword Matching (30 points)
    vectorizer = CountVectorizer().fit([resume_text, job_description])
    vectors = vectorizer.transform([resume_text, job_description])
    keyword_similarity = cosine_similarity(vectors)[0][1]
    keyword_score = int(keyword_similarity * 30)  # Max 30 points

    # Keyword Weighting (10 points)
    critical_keywords = ["python", "react", "sql", "communication", "teamwork", "leadership"]
    resume_words = resume_text.lower().split()
    critical_keyword_count = sum(1 for word in resume_words if word in critical_keywords)
    keyword_weighting_score = min(critical_keyword_count * 2, 10)  # Max 10 points

    # Section Completeness (20 points)
    sections = ["skills", "experience", "education", "projects"]
    section_score = 0
    for section in sections:
        if re.search(rf"\b{section}\b", resume_text, re.IGNORECASE):
            section_score += 5  # 5 points per section
    section_score = min(section_score, 20)  # Max 20 points

    # Section Depth (10 points)
    bullet_points = len(re.findall(r"\n\s*•", resume_text))  # Count bullet points
    section_depth_score = min(bullet_points, 10)  # Max 10 points

    # Soft Skills Detection (10 points)
    soft_skills = ["communication", "teamwork", "leadership", "problem solving", "time management"]
    soft_skill_count = sum(1 for skill in soft_skills if skill in resume_text.lower())
    soft_skill_score = min(soft_skill_count * 2, 10)  # Max 10 points

    # Education Level (10 points)
    education_levels = {
        "phd": 10,
        "master": 8,
        "bachelor": 6,
        "diploma": 4,
        "high school": 2
    }
    education_score = 0
    for level, score in education_levels.items():
        if re.search(rf"\b{level}\b", resume_text, re.IGNORECASE):
            education_score = max(education_score, score)  # Take the highest level
    education_score = min(education_score, 10)  # Max 10 points

    # Certifications (10 points)
    certifications = ["aws certified", "pmp", "scrum master", "google cloud", "microsoft certified"]
    certification_count = sum(1 for cert in certifications if cert in resume_text.lower())
    certification_score = min(certification_count * 2, 10)  # Max 10 points

    # Grammar and Readability (10 points)
    tool = language_tool_python.LanguageTool('en-US')
    grammar_errors = len(tool.check(resume_text))
    grammar_score = max(0, 10 - grammar_errors)  # Deduct 1 point per error, minimum 0

    # Customization (10 points)
    customization_score = 10 if keyword_similarity >= 0.5 else 5  # Tailored for the job

    # Total ATS Score
    ats_score = (
        keyword_score + keyword_weighting_score + section_score + 
        section_depth_score + soft_skill_score + education_score + 
        certification_score + grammar_score + customization_score
    )
    return ats_score
# Function to read PDF
def pdf_reader(file):
    return extract_text(file, laparams=LAParams())

# Function to show PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to recommend courses
def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

# Function to fetch YouTube video title
def fetch_yt_video(link):
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        return info.get('title', 'Unknown Title')

# Function to get table download link
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to predict field
def predict_field(resume_text):
    resume_text = resume_text.lower()
    field_keywords = {
        'Data Science': {
            'data science': 3,
            'data scientist': 3,
            'machine learning': 2,
            'deep learning': 2,
            'artificial intelligence': 2,
            'python': 1,
            'r programming': 1,
            'pandas': 1,
            'numpy': 1,
            'tensorflow': 1,
            'pytorch': 1,
            'scikit-learn': 1,
        },
        'Web Development': {
            'web development': 3,
            'web developer': 3,
            'frontend': 2,
            'backend': 2,
            'full stack': 2,
            'html': 1,
            'css': 1,
            'javascript': 1,
            'react': 1,
            'angular': 1,
            'node.js': 1,
            'django': 1,
            'flask': 1,
        },
        'Android Development': {
            'android development': 3,
            'android developer': 3,
            'kotlin': 2,
            'android studio': 2,
            'mobile development': 2,
            'java': 1,
            'xml': 1,
            'firebase': 1,
            'material design': 1,
        },
        'iOS Development': {
            'ios development': 3,
            'ios developer': 3,
            'swift': 2,
            'objective-c': 2,
            'xcode': 2,
            'mobile development': 1,
            'cocoa touch': 1,
            'core data': 1,
        },
        'UI/UX Development': {
            'ui/ux': 3,
            'user interface': 2,
            'user experience': 2,
            'figma': 2,
            'adobe xd': 2,
            'wireframe': 1,
            'prototype': 1,
            'sketch': 1,
            'interaction design': 1,
        },
    }

    field_scores = {field: 0 for field in field_keywords}
    for field, keywords in field_keywords.items():
        for keyword, weight in keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text):
                field_scores[field] += weight

    predicted_field = max(field_scores, key=field_scores.get)
    if field_scores[predicted_field] == 0:
        return 'Other'
    else:
        return predicted_field

# Function to get a new database connection
def get_db_connection():
    try:
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='sra_new'
        )
        return connection
    except pymysql.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to insert data into the new table
def insert_data(name, email, timestamp, no_of_pages, reco_field, cand_level, skills, ats_score):
    try:
        connection = get_db_connection()
        if connection is None:
            st.error("Failed to establish database connection.")
            return

        cursor = connection.cursor()
        insert_sql = """
        INSERT INTO user_data_new 
        (Name, Email_ID, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, ATS_Score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        rec_values = (
            name, 
            email, 
            timestamp, 
            str(no_of_pages), 
            reco_field, 
            cand_level, 
            skills, 
            str(ats_score)
        )
        
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        st.success("Resume data saved to the database.")
    except pymysql.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to fetch data from the database
def fetch_data():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM user_data_new")
                data = cursor.fetchall()
                return data
    except pymysql.Error as e:
        st.error(f"Database error during fetch: {e}")
        return None

# Streamlit App
st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='./Logo/new_logo.png',
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Career Recommendations", "Resume Writing Tips", "Courses Recommendations", "Skills Recommendations", "YouTube Video Recommendations", "Placement Prediction", "Resume Builder", "Admin"])

import re

# Function to extract email using regex
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

# Function to extract phone number using regex
def extract_phone_number(text):
    # Comprehensive regex pattern for phone numbers
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
    phones = re.findall(phone_pattern, text)
    # Filter out invalid matches (e.g., single digits)
    valid_phones = [phone for phone in phones if len(re.sub(r'[^0-9]', '', phone)) >= 7]
    return valid_phones[0] if valid_phones else None

# Function to extract name
def extract_name(text):
    # Try spaCy's NER first
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    
    # If NER fails, assume the first line is the name
    first_line = text.split('\n')[0].strip()
    return first_line if first_line else None


# Function to extract skills using a predefined list
def extract_skills(text):
    skills_list = [
        'python', 'java', 'machine learning', 'data analysis', 'html', 'css', 
        'javascript', 'react', 'node.js', 'sql', 'communication', 'operating system', 
        'backend', 'express.js', 'gui', 'css', 'react', 'backend', 'express.js', 
        'sql', 'operating system', 'communication'
    ]
    skills = []
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            skills.append(skill)
    return skills

# Function to parse resume text
def parse_resume(resume_text):
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone_number = extract_phone_number(resume_text)
    skills = extract_skills(resume_text)
    return {
        'name': name,
        'email': email,
        'phone_number': phone_number,
        'skills': skills
    }
# Function to calculate candidate level based on resume length
def calculate_candidate_level(resume_text):
    # Approximate number of pages (500 words per page)
    no_of_pages = len(resume_text.split()) // 500
    if no_of_pages == 1:
        return "Fresher", no_of_pages
    elif no_of_pages == 2:
        return "Intermediate", no_of_pages
    elif no_of_pages >= 3:
        return "Experienced", no_of_pages
    else:
        return "Unknown", no_of_pages


# Example usage in the Home Page
if page == "Home":
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 2s;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center;">Smart Resume Analyzer</h1>
    </div>
    """, unsafe_allow_html=True)

    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="home-animation")

    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if pdf_file is not None:
        save_image_path = './Uploaded_Resumes/' + pdf_file.name
        with open(save_image_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        show_pdf(save_image_path)
        try:
            # Extract text from PDF
            resume_text = pdf_reader(save_image_path)

            # Parse the resume text
            parsed_data = parse_resume(resume_text)
            cand_level, no_of_pages = calculate_candidate_level(resume_text)

            # Display extracted data
            st.header("**Resume Analysis**")
            st.success("Hello " + (parsed_data['name'] if parsed_data['name'] else "User"))
            st.subheader("**Your Basic info**")
            st.text(f'Name: {parsed_data["name"]}')
            st.text(f'Email: {parsed_data["email"]}')
            st.text(f'Contact: {parsed_data["phone_number"]}')
            st.text(f'Skills: {", ".join(parsed_data["skills"])}')
            predicted_field = predict_field(resume_text)
            st.text(f'Predicted Field: {predicted_field}')

            # Display candidate level feedback
            if cand_level == "Fresher":
                st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>Oh!!!! You are a Fresher!.</h4>''', unsafe_allow_html=True)
            elif cand_level == "Intermediate":
                st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Nice! Intermediate level!</h4>''', unsafe_allow_html=True)
            elif cand_level == "Experienced":
                st.markdown('''<h4 style='text-align: left; color: #fba171;'>Wow!! You are pretty experienced!!</h4>''', unsafe_allow_html=True)
          
            # ATS Score Checker
            if st.button("Check ATS Score"):
                job_description = st.text_area("Paste the job description here")
                if job_description:
                    ats_score = calculate_ats_score(resume_text, job_description)
                    st.subheader("ATS Score")
                    st.progress(ats_score)
                    st.success(f"Your ATS Score: {ats_score}/100")

                    st.subheader("Feedback")
                    if ats_score >= 80:
                        st.success("Great job! Your resume is well-optimized for ATS.")
                    elif ats_score >= 50:
                        st.warning("Your resume is decent but could use some improvements.")
                    else:
                        st.error("Your resume needs significant improvements to pass ATS filters.")

                    st.subheader("Tips to Improve Your ATS Score")
                    st.write("1. **Use Keywords**: Include keywords from the job description.")
                    st.write("2. **Format Properly**: Use standard fonts, headings, and bullet points.")
                    st.write("3. **Tailor Your Resume**: Customize your resume for each job application.")
                    st.write("4. **Keep It Concise**: Aim for 1-2 pages.")

                    insert_data(
                        name=parsed_data['name'],
                        email=parsed_data['email'],
                        timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                        no_of_pages=no_of_pages,
                        reco_field=predicted_field,
                        cand_level=cand_level,
                        skills=', '.join(parsed_data['skills']),
                        ats_score=str(ats_score)
                    )
        except Exception as e:
            st.error(f"Error parsing resume: {e}")

# Other pages (Career Recommendations, Resume Writing Tips, Courses Recommendations, Skills Recommendations, YouTube Video Recommendations, Admin) remain the same as in your original code.
elif page == "Career Recommendations":
    st.title("🎯 Career Recommendations")
    st.write("Here are some personalized career recommendations based on your resume analysis:")

    # Fetch user data from the database
    user_data = fetch_data()
    if user_data:
        # Convert user data to a DataFrame
        df = pd.DataFrame(user_data, columns=['ID', 'Name', 'Email', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'ATS Score'])
        
        # Get the latest user entry (assuming the last entry is the most recent)
        latest_user = df.iloc[-1]
        
        # Extract relevant fields
        predicted_field = latest_user['Predicted Field']
        user_level = latest_user['User Level']
        skills = latest_user['Actual Skills']
        ats_score = latest_user['ATS Score']

        # Display user information in a visually appealing way
        st.subheader("👤 Your Profile Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🎯 Predicted Field**")
            st.info(predicted_field)
        with col2:
            st.markdown("**📊 Experience Level**")
            st.success(user_level)
        with col3:
            st.markdown("**📝 ATS Score**")
            st.progress(int(ats_score) / 100)
            st.caption(f"{ats_score}/100")

        st.markdown("**🛠️ Key Skills**")
        st.write(skills)

        # Career Recommendations Logic
        st.subheader("🚀 Career Recommendations")
        st.write("Based on your skills and predicted field, here are some career options you can explore:")

        # Create a grid layout for career cards
        col1, col2, col3 = st.columns(3)

        if predicted_field == "Data Science":
            with col1:
                st.markdown("**📊 Data Scientist**")
                st.write("Work with data to build predictive models and derive insights.")
                st.markdown("**Skills:** Python, Machine Learning, Data Analysis")
            with col2:
                st.markdown("**🤖 Machine Learning Engineer**")
                st.write("Design and implement machine learning systems.")
                st.markdown("**Skills:** TensorFlow, PyTorch, Deep Learning")
            with col3:
                st.markdown("**📈 Data Analyst**")
                st.write("Analyze data to help businesses make informed decisions.")
                st.markdown("**Skills:** SQL, Excel, Data Visualization")

        elif predicted_field == "Web Development":
            with col1:
                st.markdown("**🌐 Frontend Developer**")
                st.write("Build user interfaces and interactive web applications.")
                st.markdown("**Skills:** HTML, CSS, JavaScript")
            with col2:
                st.markdown("**🔙 Backend Developer**")
                st.write("Develop server-side logic and databases.")
                st.markdown("**Skills:** Node.js, Python, Django")
            with col3:
                st.markdown("**🚀 Full Stack Developer**")
                st.write("Work on both frontend and backend development.")
                st.markdown("**Skills:** React, MongoDB, REST APIs")

        elif predicted_field == "Android Development":
            with col1:
                st.markdown("**📱 Android Developer**")
                st.write("Build Android applications using Kotlin or Java.")
                st.markdown("**Skills:** Kotlin, Android Studio, XML")
            with col2:
                st.markdown("**📲 Mobile App Developer**")
                st.write("Develop cross-platform mobile applications.")
                st.markdown("**Skills:** Flutter, React Native, Dart")
            with col3:
                st.markdown("**🛠️ Mobile Application Architect**")
                st.write("Design scalable mobile app architectures.")
                st.markdown("**Skills:** System Design, Cloud Integration")

        else:
            st.write("Based on your skills, here are some general career options:")
            with col1:
                st.markdown("**💻 Software Engineer**")
                st.write("Develop software applications and systems.")
                st.markdown("**Skills:** Python, Java, C++")
            with col2:
                st.markdown("**📊 Project Manager**")
                st.write("Manage software projects and teams.")
                st.markdown("**Skills:** Agile, Scrum, Communication")
            with col3:
                st.markdown("**📝 Technical Writer**")
                st.write("Create technical documentation and guides.")
                st.markdown("**Skills:** Writing, Markdown, API Documentation")

        # Additional Recommendations Based on ATS Score
        st.subheader("📈 Tips to Improve Your Career Prospects")
        if int(ats_score) >= 80:
            st.success("🎉 Your resume is well-optimized! Keep applying to roles that match your skills.")
        elif int(ats_score) >= 50:
            st.warning("✨ Your resume is decent, but consider improving it to stand out more.")
            st.write("- **Tailor your resume** for each job application.")
            st.write("- **Highlight your achievements** with quantifiable results.")
            st.write("- **Learn new skills** relevant to your field.")
        else:
            st.error("🚨 Your resume needs significant improvements to increase your chances.")
            st.write("- **Focus on keyword optimization** for Applicant Tracking Systems (ATS).")
            st.write("- **Improve formatting** to make your resume more readable.")
            st.write("- **Gain more experience** through internships or freelance projects.")

    else:
        st.warning("⚠️ No user data found. Please upload and analyze your resume first.")

elif page == "Resume Writing Tips":
    st.title("📝 Resume Writing Tips")
    st.write("Here are some actionable tips to help you create a standout resume:")

    # Tip 1: Use a Clear and Professional Format
    st.subheader("🎨 1. Use a Clear and Professional Format")
    with st.expander("Learn More"):
        st.write("""
        - **Choose a clean, easy-to-read font** like Arial, Calibri, or Times New Roman.
        - **Use consistent formatting** for headings, bullet points, and sections.
        - **Avoid clutter** by leaving enough white space.
        - **Stick to 1-2 pages** unless you have extensive experience.
        """)

    st.markdown("---")

    # Tip 2: Tailor Your Resume for the Job
    st.subheader("🎯 2. Tailor Your Resume for the Job")
    with st.expander("Learn More"):
        st.write("""
        - **Customize your resume** for each job application.
        - **Highlight relevant skills and experiences** that match the job description.
        - **Use keywords** from the job posting to pass Applicant Tracking Systems (ATS).
        """)

    st.markdown("---")

    # Tip 3: Write a Strong Summary or Objective
    st.subheader("📄 3. Write a Strong Summary or Objective")
    with st.expander("Learn More"):
        st.write("""
        - **Summarize your career goals and key qualifications** in 2-3 sentences.
        - **Focus on what you can offer** to the employer, not just what you want.
        - **Example**: "Results-driven marketing professional with 5+ years of experience in digital campaigns and brand management."
        """)

    st.markdown("---")

    # Tip 4: Highlight Your Achievements
    st.subheader("🏆 4. Highlight Your Achievements")
    with st.expander("Learn More"):
        st.write("""
        - **Use action verbs** like "achieved," "developed," "led," and "optimized."
        - **Quantify your accomplishments** with numbers, percentages, or timeframes.
        - **Example**: "Increased website traffic by 40% through SEO optimization."
        """)

    st.markdown("---")

    # Tip 5: Include Relevant Skills
    st.subheader("🛠️ 5. Include Relevant Skills")
    with st.expander("Learn More"):
        st.write("""
        - **List both hard skills (technical)** and **soft skills (interpersonal)**.
        - **Match your skills** to the job description.
        - **Example**: "Proficient in Python, SQL, and data visualization tools like Tableau."
        """)

    st.markdown("---")

    # Tip 6: Proofread and Edit
    st.subheader("🔍 6. Proofread and Edit")
    with st.expander("Learn More"):
        st.write("""
        - **Check for spelling and grammar errors**.
        - **Ask a friend or mentor** to review your resume.
        - **Use tools like Grammarly** to catch mistakes.
        """)

    st.markdown("---")

    # Bonus: Resume Templates
    st.subheader("🎁 Bonus: Resume Templates")
    with st.expander("Learn More"):
        st.write("""
        - Use professional resume templates to save time and ensure a polished look.
        - You can find free templates on platforms like:
            - [Canva](https://www.canva.com/resumes/templates/)
            - [Google Docs](https://docs.google.com/document/u/0/?ftv=1&tgif=c)
            - [LinkedIn](https://www.linkedin.com/resume-builder/)
        """)

elif page == "Courses Recommendations":
    st.title("🎓 Courses Recommendations")
    st.write("Here are some carefully curated courses to help you enhance your skills and advance your career:")

    # Define course categories
    course_categories = {
        "Data Science": [
            {"name": "Python for Data Science", "link": "https://www.coursera.org/learn/python-for-data-science"},
            {"name": "Machine Learning by Andrew Ng", "link": "https://www.coursera.org/learn/machine-learning"},
            {"name": "Data Visualization with Tableau", "link": "https://www.coursera.org/learn/data-visualization-tableau"},
        ],
        "Web Development": [
            {"name": "HTML, CSS, and JavaScript", "link": "https://www.coursera.org/learn/html-css-javascript"},
            {"name": "React for Beginners", "link": "https://www.udemy.com/course/react-for-beginners/"},
            {"name": "Full Stack Web Development", "link": "https://www.coursera.org/learn/full-stack-web-development"},
        ],
        "Android Development": [
            {"name": "Android App Development", "link": "https://www.udacity.com/course/android-developer-nanodegree-by-google--nd801"},
            {"name": "Kotlin for Android", "link": "https://www.udemy.com/course/kotlin-for-android-development/"},
            {"name": "Flutter for Beginners", "link": "https://www.udemy.com/course/flutter-bootcamp-with-dart/"},
        ],
        "UI/UX Design": [
            {"name": "UI/UX Design Fundamentals", "link": "https://www.coursera.org/learn/ui-ux-design-fundamentals"},
            {"name": "Figma for Beginners", "link": "https://www.udemy.com/course/figma-for-beginners/"},
            {"name": "Adobe XD Essentials", "link": "https://www.udemy.com/course/adobe-xd-essentials/"},
        ],
    }

    # Display course recommendations in a grid layout
    for category, courses in course_categories.items():
        st.subheader(f"📚 {category}")
        cols = st.columns(3)  # Create 3 columns for each category
        for i, course in enumerate(courses):
            with cols[i % 3]:
                st.markdown(f"""
                **{course['name']}**  
                [🔗 Enroll Now]({course['link']})
                """)
        st.markdown("---")  # Add a divider between categories

    # Additional Resources Section
    st.subheader("📖 Additional Resources")
    st.write("Explore more resources to boost your learning:")
    st.markdown("""
    - **[Coursera](https://www.coursera.org)**: Online courses from top universities.  
    - **[Udemy](https://www.udemy.com)**: Affordable courses on a wide range of topics.  
    - **[edX](https://www.edx.org)**: Free and paid courses from institutions like MIT and Harvard.  
    - **[Khan Academy](https://www.khanacademy.org)**: Free courses on programming and more.  
    """)

    # Call to Action
    st.markdown("---")
    st.write("🚀 Ready to take your skills to the next level? Start learning today!")

elif page == "Skills Recommendations":
    st.title("🛠️ Skills Recommendations")
    st.write("Here are some skills you should consider adding to your resume based on your predicted field:")

    # Fetch user data from the database
    user_data = fetch_data()
    if user_data:
        # Convert user data to a DataFrame
        df = pd.DataFrame(user_data, columns=['ID', 'Name', 'Email', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'ATS Score'])
        
        # Get the latest user entry (assuming the last entry is the most recent)
        latest_user = df.iloc[-1]
        
        # Extract the predicted field
        predicted_field = latest_user['Predicted Field']

        # Define skill recommendations for each field
        skill_recommendations = {
            "Data Science": [
                "Python",
                "Machine Learning",
                "Data Visualization",
                "SQL",
                "Pandas",
                "NumPy",
                "TensorFlow",
                "PyTorch",
                "Scikit-learn",
                "Data Analysis",
            ],
            "Web Development": [
                "HTML",
                "CSS",
                "JavaScript",
                "React",
                "Node.js",
                "Django",
                "Flask",
                "REST APIs",
                "MongoDB",
                "Git",
            ],
            "Android Development": [
                "Kotlin",
                "Java",
                "Android Studio",
                "Firebase",
                "Material Design",
                "XML",
                "Flutter",
                "Mobile App Development",
                "REST APIs",
                "Git",
            ],
            "UI/UX Design": [
                "Figma",
                "Adobe XD",
                "Wireframing",
                "Prototyping",
                "User Research",
                "Interaction Design",
                "Visual Design",
                "Usability Testing",
                "Sketch",
                "InVision",
            ],
            "Other": [
                "Communication",
                "Problem Solving",
                "Project Management",
                "Team Collaboration",
                "Time Management",
                "Leadership",
                "Critical Thinking",
                "Adaptability",
                "Creativity",
                "Technical Writing",
            ],
        }

        # Display skills based on the predicted field
        if predicted_field in skill_recommendations:
            st.subheader(f"📚 Recommended Skills for {predicted_field}")
            cols = st.columns(3)  # Create 3 columns for displaying skills
            for i, skill in enumerate(skill_recommendations[predicted_field]):
                with cols[i % 3]:
                    st.markdown(f"""
                    **{skill}**  
                    """)
            st.markdown("---")  # Add a divider

            # Additional Resources Section
            st.subheader("📖 Additional Resources")
            st.write(f"Explore more resources to learn and master {predicted_field} skills:")
            st.markdown("""
            - **[Coursera](https://www.coursera.org)**: Online courses from top universities.  
            - **[Udemy](https://www.udemy.com)**: Affordable courses on a wide range of topics.  
            - **[edX](https://www.edx.org)**: Free and paid courses from institutions like MIT and Harvard.  
            - **[Khan Academy](https://www.khanacademy.org)**: Free courses on programming and more.  
            """)

            # Call to Action
            st.markdown("---")
            st.write("🚀 Ready to take your skills to the next level? Start learning today!")
        else:
            st.warning("No predicted field found. Please upload and analyze your resume first.")
    else:
        st.warning("No user data found. Please upload and analyze your resume first.")

elif page == "YouTube Video Recommendations":
    st.title("🎥 YouTube Video Recommendations")
    st.write("Here are some curated YouTube videos to help you with your career:")

    # Fetch user data from the database
    user_data = fetch_data()
    if user_data:
        # Convert user data to a DataFrame
        df = pd.DataFrame(user_data, columns=['ID', 'Name', 'Email', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'ATS Score'])
        
        # Get the latest user entry (assuming the last entry is the most recent)
        latest_user = df.iloc[-1]
        
        # Extract the predicted field
        predicted_field = latest_user['Predicted Field']

        # Define YouTube video links for each field
        video_links = {
            "Data Science": ds_videos,
            "Web Development": web_videos,
            "Android Development": android_videos,
            "iOS Development": ios_videos,
            "UI/UX Development": uiux_videos,
        }

        # Display video recommendations based on the predicted field
        if predicted_field in video_links:
            st.subheader(f"📚 Recommended YouTube Videos for {predicted_field}")
            cols = st.columns(2)  # Create 2 columns for displaying videos
            for i, video_link in enumerate(video_links[predicted_field]):
                with cols[i % 2]:
                    st.video(video_link)
            st.markdown("---")  # Add a divider

            # Additional Resources Section
            st.subheader("📖 Additional Resources")
            st.write(f"Explore more resources to learn and master {predicted_field}:")
            st.markdown("""
            - **[Coursera](https://www.coursera.org)**: Online courses from top universities.  
            - **[Udemy](https://www.udemy.com)**: Affordable courses on a wide range of topics.  
            - **[edX](https://www.edx.org)**: Free and paid courses from institutions like MIT and Harvard.  
            - **[Khan Academy](https://www.khanacademy.org)**: Free courses on programming and more.  
            """)

            # Call to Action
            st.markdown("---")
            st.write("🚀 Ready to take your skills to the next level? Start learning today!")
        else:
            st.warning("No predicted field found. Please upload and analyze your resume first.")
    else:
        st.warning("No user data found. Please upload and analyze your resume first.")

elif page == "Placement Prediction":
    import pickle
    import numpy as np
    import streamlit as st
    from streamlit_lottie import st_lottie
    import requests

    # Load Lottie animation
    def load_lottie_url(url):
        response = requests.get(url)
        if response.status_code != 200:
            return None
        return response.json()

    lottie_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")

    # Load the trained model
    try:
        model = pickle.load(open("placement_model.pkl", "rb"))
    except FileNotFoundError:
        st.error("Placement prediction model not found! Train and save 'placement_model.pkl' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

    # Placement Prediction Section
    st.title("📊 Placement Prediction")

    # Display Lottie animation
    if lottie_animation:
        st_lottie(lottie_animation, height=200, key="placement-animation")

    st.markdown("""
    <style>
    .stNumberInput > div > input {
        font-size: 18px;
        padding: 10px;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    # Input fields
    st.subheader("Enter Your Details")
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")
    with col2:
        internships = st.number_input("Number of Internships:", min_value=0, max_value=10, step=1)

    col3, col4 = st.columns(2)
    with col3:
        projects = st.number_input("Number of Projects:", min_value=0, max_value=20, step=1)
    with col4:
        ats_score = st.slider("ATS Score (out of 100):", 0, 100, 50)

    # Predict button
    if st.button("Predict Placement Chances"):
        try:
            # Validate input (allow 0 for internships and projects)
            if cgpa < 0 or cgpa > 10 or internships < 0 or internships > 10 or projects < 0 or projects > 20 or ats_score < 0 or ats_score > 100:
                st.warning("⚠️ Please enter valid values in all fields.")
            else:
                input_data = np.array([[cgpa, internships, projects, ats_score]])
                
                # Make prediction
                prediction = model.predict_proba(input_data)[0][1]  # Probability of getting placed

                # Display prediction
                st.success(f"📌 Your Chances of Getting Placed: **{prediction * 100:.2f}%**")
                
                # Feedback based on prediction
                if prediction >= 0.8:
                    st.success("🔥 Excellent! High chances of placement.")
                    st.markdown("""
                    **Tips to Maintain Your Edge:**
                    - Continue building your skills.
                    - Network with industry professionals.
                    - Prepare for interviews with mock sessions.
                    """)
                elif prediction >= 0.5:
                    st.warning("⚠️ Good, but improve your resume and skills.")
                    st.markdown("""
                    **Tips to Improve:**
                    - Add more projects to your portfolio.
                    - Gain additional internships or certifications.
                    - Tailor your resume for specific job roles.
                    """)
                else:
                    st.error("❌ Low chances. Gain more experience & skills.")
                    st.markdown("""
                    **Tips to Improve:**
                    - Focus on improving your CGPA.
                    - Participate in hackathons and coding competitions.
                    - Seek internships to gain practical experience.
                    """)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif page == "Resume Builder":
    st.write("Build your resume")


# Admin Page
elif page == "Admin":
    st.title("Admin Side")
    st.sidebar.markdown("**Admin Login**")
    ad_user = st.sidebar.text_input("Username")
    ad_password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button('Login'):
        if ad_user == 'admin' and ad_password == 'admin123':
            st.success("Welcome Admin")

            # Fetch and display user data
            user_data = fetch_data()
            if user_data:
                st.header("**User's Data**")
                df = pd.DataFrame(user_data, columns=['ID', 'Name', 'Email', 'Timestamp', 'Total Page', 'Predicted Field', 'User Level', 'Actual Skills', 'ATS Score'])
                st.dataframe(df)

                # Download User Data
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)

                # Pie Chart for Predicted Field Recommendations
                st.subheader("📈 **Pie-Chart for Predicted Field Recommendations**")
                labels = df['Predicted Field'].unique()
                values = df['Predicted Field'].value_counts()
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                # Pie Chart for User's Experience Level
                st.subheader("📈 **Pie-Chart for User's Experience Level**")
                labels = df['User Level'].unique()
                values = df['User Level'].value_counts()
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's Experience Level")
                st.plotly_chart(fig)
            else:
                st.warning("No user data found.")
        else:
            st.error("Wrong ID & Password Provided")

# Run the app
if __name__ == "__main__":
    pass