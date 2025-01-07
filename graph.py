import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results_with_classifications.csv')

# Count the occurrences of each label
label_counts = df['labels'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Labels')
plt.show()

# Word Cloud for Answer Text
from wordcloud import WordCloud

text = ' '.join(df['answer'])
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Answer Text')
plt.show()

