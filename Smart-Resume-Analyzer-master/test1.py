import streamlit as st
import nltk
import spacy
from streamlit_lottie import st_lottie
import requests
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import pafy
import plotly.express as px
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
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.error("The 'en_core_web_sm' model is not installed. Please run `python -m spacy download en_core_web_sm` in your terminal.")
    st.stop()

# Function to load Lottie animations
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load Lottie animation
lottie_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")

# Function to calculate ATS score
def calculate_ats_score(resume_text, job_description):
    # Keyword Matching (30 points)
    vectorizer = CountVectorizer().fit([resume_text, job_description])
    vectors = vectorizer.transform([resume_text, job_description])
    keyword_similarity = cosine_similarity(vectors)[0][1]
    keyword_score = int(keyword_similarity * 30)  # Max 30 points

    # Formatting (20 points)
    formatting_score = 0
    if re.search(r"\b(?:skills|experience|education)\b", resume_text, re.IGNORECASE):
        formatting_score += 10  # Points for having key sections
    if len(re.findall(r"\n\s*•", resume_text)) >= 5:  # Check for bullet points
        formatting_score += 10

    # Length (10 points)
    length_score = 10 if 500 <= len(resume_text.split()) <= 1000 else 5  # Ideal length: 1-2 pages

    # Customization (20 points)
    customization_score = 20 if keyword_similarity >= 0.5 else 10  # Tailored for the job

    # Total ATS Score
    ats_score = keyword_score + formatting_score + length_score + customization_score
    return ats_score

# Function to read PDF
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

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

import re

def predict_field(resume_text):
    # Convert resume text to lowercase for case-insensitive matching
    resume_text = resume_text.lower()

    # Define keywords and their weights for each field
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

    # Calculate scores for each field
    field_scores = {field: 0 for field in field_keywords}
    for field, keywords in field_keywords.items():
        for keyword, weight in keywords.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', resume_text):
                field_scores[field] += weight

    # Get the field with the highest score
    predicted_field = max(field_scores, key=field_scores.get)

    # If no significant match, return 'Other'
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
        # Establish a new database connection
        connection = get_db_connection()
        if connection is None:
            st.error("Failed to establish database connection.")
            return

        cursor = connection.cursor()

        # Define the SQL query with explicit column names
        insert_sql = """
        INSERT INTO user_data_new 
        (Name, Email_ID, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, ATS_Score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Prepare the values to be inserted
        rec_values = (
            name, 
            email, 
            timestamp, 
            str(no_of_pages), 
            reco_field, 
            cand_level, 
            skills, 
            str(ats_score)  # Convert ATS_Score to string if necessary
        )
        
        # Debug: Print the SQL query and values
        st.write("Executing SQL:", insert_sql)
        st.write("With values:", rec_values)
        
        # Execute the query
        cursor.execute(insert_sql, rec_values)
        connection.commit()
        st.success("Resume data saved to the database.")
    except pymysql.Error as e:
        st.error(f"Database error: {e}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Function to fetch data from the database
def fetch_data():
    try:
        # Use a context manager for the connection and cursor
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
page = st.sidebar.radio("Go to", ["Home", "Career Recommendations", "Resume Writing Tips", "Courses Recommendations", "Skills Recommendations", "YouTube Video Recommendations","Placement Prediction", "Admin"])

# Home Page
if page == "Home":
    # Add custom CSS for animations
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

    # Title with animation
    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center;">Smart Resume Analyzer</h1>
    </div>
    """, unsafe_allow_html=True)

    # Display Lottie animation
    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="home-animation")

    # Upload Resume
    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if pdf_file is not None:
        save_image_path = './Uploaded_Resumes/' + pdf_file.name
        with open(save_image_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        show_pdf(save_image_path)
        try:
            # Parse the resume
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            
            # Debug: Print the extracted resume data
            st.write("Extracted Resume Data:", resume_data)
            
            if resume_data:
                resume_text = pdf_reader(save_image_path)

                st.header("**Resume Analysis**")
                st.success("Hello " + resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: ' + resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: ' + str(resume_data['no_of_pages']))
                    st.text('Skills: ' + ', '.join(resume_data['skills']))
                    predicted_field = predict_field(resume_text)
                    st.text('Predicted Field: ' + predicted_field)
                except Exception as e:
                    st.error(f"Error displaying basic info: {e}")

                # Determine candidate level
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>Oh!!!! You are a Fresher!.</h4>''', unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Nice! Intermediate level!</h4>''', unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>Wow!! You are pretty experienced!!''', unsafe_allow_html=True)

                # ATS Score Checker (Optional)
                if st.button("Check ATS Score"):
                    job_description = st.text_area("Paste the job description here")
                    if job_description:
                        ats_score = calculate_ats_score(resume_text, job_description)
                        st.subheader("ATS Score")
                        st.progress(ats_score)
                        st.success(f"Your ATS Score: {ats_score}/100")

                        # Feedback
                        st.subheader("Feedback")
                        if ats_score >= 80:
                            st.success("Great job! Your resume is well-optimized for ATS.")
                        elif ats_score >= 50:
                            st.warning("Your resume is decent but could use some improvements.")
                        else:
                            st.error("Your resume needs significant improvements to pass ATS filters.")

                        # Tips to Improve
                        st.subheader("Tips to Improve Your ATS Score")
                        st.write("1. **Use Keywords**: Include keywords from the job description.")
                        st.write("2. **Format Properly**: Use standard fonts, headings, and bullet points.")
                        st.write("3. **Tailor Your Resume**: Customize your resume for each job application.")
                        st.write("4. **Keep It Concise**: Aim for 1-2 pages.")

                        # Insert data into the database
                        insert_data(
                            name=resume_data['name'],
                            email=resume_data['email'],
                            timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
                            no_of_pages=resume_data['no_of_pages'],
                            reco_field=predicted_field,
                            cand_level=cand_level,
                            skills=', '.join(resume_data['skills']),
                            ats_score=str(ats_score)
                        )
        except Exception as e:
            st.error(f"Error parsing resume: {e}")

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