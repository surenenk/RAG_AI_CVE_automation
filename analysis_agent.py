import pandas as pd
import openai

openai.api_key = "your_api_key_here"


def analyze_cves():
    df = pd.read_csv("cve_data.csv")
    critical_cves = df[df["CVSS_Score"] >= 7].head(10)  # Get top 10 high-risk CVEs

    prompt = "Analyze the following high-risk vulnerabilities:\n"
    for _, row in critical_cves.iterrows():
        prompt += f"\nCVE: {row['CVE_ID']} - {row['Description']} (Score: {row['CVSS_Score']})"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a cybersecurity expert."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("📊 AI Security Analysis:")
    print(analyze_cves())
