import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#Load the data
stData=pd.read_csv("C:/Users/feelh/OneDrive/Desktop/StudentsData.csv")

#Save data
stData.to_csv('StudentData',index=False)
#----------------Preprocessing----------------

#   1.  Handle missing values by removing rows which has NULL values

print('Number of rows before handling na: ',len(stData))
stData.dropna(inplace=True)
print('Number of rows after handling na: ',len(stData))
#inplace=True means the operation is applied directly to the original DataFrame (no copy is created).


#   2.  Create a new feature (Feature Creation) from the field Absence called 'Attendance' ((Total Classes-Absences)/ Total classes)×100
stData['Attendance']=round((170-stData['Absences'])/170,2)*100

#   3.  Delete useless fields (reduce data)
stData=stData.drop(columns=['Ethnicity','ParentalEducation','Tutoring','ParentalSupport','Extracurricular','Sports','Music','Volunteering','GradeClass'])

#   4.  Round StudyTimeWeekly and GPA fields
stData['StudyTimeWeekly']=round(stData['StudyTimeWeekly'],1)
stData['GPA']=round(stData['GPA'],2)

#   5.  Convert values in 'StudentID' into str instead of int
stData['StudentID']=stData['StudentID'].astype(str)

#   6.  Encoding Final Grade column into numerical values A=5, B=4, C=3, D=2, F=1
stData['Final grade']=stData['Final grade'].replace({'A':5,'B':4,'C':3,'D':2,'F':1}).astype(int)

#   7.  Replace 0 with 'F' and 1 with 'M' for Gender field
stData['Gender']=stData['Gender'].replace({1:'M',0:'F'})


#   8.  split 'Test scores (Math, Reading, Writing)' column into three columns Math, Reading and Writing
stData[['MathScore','ReadingScore', 'WritingScore']]=stData['Test scores (Math, Reading, Writing)'].str.split(',',expand=True).astype(int)
# expand=True    Expands them into a DataFrame (3 columns).


#   9.  Drop 'Test scores (Math, Reading, Writing)' column
stData=stData.drop(columns=['Test scores (Math, Reading, Writing)'])

#   10. Add field TotalScores
stData['TotalScores']=stData[['MathScore','ReadingScore','WritingScore']].sum(axis=1)
# axis=1    means: For each row, sum the values of MathScore, ReadingScore, and WritingScore

#axis=0 → Operate down the rows, i.e., column-wise (default).
#axis=1 → Operate across the columns, i.e., row-wise.


#   11. Rename Column  'Internet access at home (Yes/No)' into 'InternetAccess' and 'Participation in class' into 'Participation'
stData.rename(columns={'Internet access at home (Yes/No)':'InternetAccess'},inplace=True)
stData.rename(columns={'Participation in class':'Participation'},inplace=True)
#inplace=True modifies the DataFrame directly; if you omit it, rename() returns a new DataFrame.

#   12. save the data in csv file in pyCharm
#index=False → Removes row index from the file.
stData.to_csv('StudentData',index=False)

#   13. Convert it to DataFrame
stData=pd.DataFrame(stData)

#----------------Descriptive Analysis----------------


#   1.  Summary statistics (mean, median, mode, etc.)
print(stData[['Age','StudyTimeWeekly','GPA','Final grade','Attendance','MathScore','ReadingScore','WritingScore']].describe())

#   2.  Visualize distributions of grades, attendance, and study time.
stData[['MathScore','ReadingScore','WritingScore','Attendance','StudyTimeWeekly']].hist(bins=10, figsize=(12,12))
plt.suptitle('Distributions of Grades, Attendance, and Study Time')
plt.show()


#----------------Diagnostic Analytics----------------

#   1.  The relation between StudyTimeWeekly and Final grade
#Find the correlation between them
TimeGradeCorr=stData['StudyTimeWeekly'].corr(stData['Final grade'])
print('Correlation between "StudyTimeWeekly" and "Final grade":',TimeGradeCorr)

#   2.  Find the correlation between Attendance and GPA
AttendGPACorr=stData['Attendance'].corr(stData['GPA'])
print('Correlation between "Attendance" and "GPA":',AttendGPACorr)

#   3.  Gender-based differences by visulization

# Calculate average GPA per gender
avg_gpa = stData.groupby('Gender')['GPA'].mean()
print(avg_gpa)

# Plot them
plt.bar(avg_gpa.index, avg_gpa.values, color=['pink','cyan'])
plt.title('Average GPA by Gender')
plt.xlabel('Gender')
plt.ylabel('Average GPA')
plt.show()

#----------------Predictive Analytics----------------
# 1.    Use simple linear regression to predict final grade based on study hours=20 or attendance=90.
x=np.array(stData['StudyTimeWeekly']).reshape(-1,1)
y=np.array(stData['Final grade'])
plt.title('Relationship between Study Time and Final Grade')
plt.xlabel('Study Time Weekly (hours)')
plt.ylabel('Final Grade')
plt.scatter(x,y)
plt.show()
model=LinearRegression()
model.fit(x,y)

print("Predicted final grade for study hours=20: ",model.predict(np.array([20]).reshape(-1,1)))

# ----------------Prescriptive Insights----------------
# Suggest strategies (e.g., “Students who study > 10 hours/week score 20% higher on
# average.”)

#check the relationship between StudyTimeWeekly and TotalScores
print('the relationship between StudyTimeWeekly and TotalScores: ',stData[['StudyTimeWeekly', 'TotalScores']].corr())

# Create groups based on study hours
bins = [0, 5, 10, 15, 20]
labels = ['0-5 hrs', '6-10 hrs', '11-15 hrs', '16-20 hrs']
stData['StudyGroup'] = pd.cut(stData['StudyTimeWeekly'], bins=bins, labels=labels)

# Calculate average score for each group
avg_score = stData.groupby('StudyGroup')['TotalScores'].mean()
print(avg_score)

# Find the highest performing group
best_group = avg_score.idxmax()
best_score = avg_score.max()

print(f"Students who study in the range {best_group} have the highest average score: {best_score:.2f}")

#You can generalize
for group, score in avg_score.items():
    print(f"Students studying {group} hours per week score {score:.1f} on average.")

#Visualize insight

avg_score.plot(kind='bar', color='skyblue')
plt.title('Average Score by Study Hours Group')
plt.ylabel('Average Score')
plt.xlabel('Study Hours Group')
plt.show()
