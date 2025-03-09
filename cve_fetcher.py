import requests
import pandas as pd

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_cve_data():
    params = {"resultsPerPage": 50}  # Fetch latest 50 vulnerabilities
    response = requests.get(NVD_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    return None


def save_cve_to_csv():
    data = fetch_cve_data()
    if data:
        cve_list = []
        for item in data.get("vulnerabilities", []):
            cve_id = item["cve"]["id"]
            description = item["cve"]["descriptions"][0]["value"]
            cvss_score = item["cve"]["metrics"].get("cvssMetricV31", [{}])[0].get("cvssData", {}).get("baseScore", 0)
            status = "Unpatched" if "exploit" in description.lower() else "Patched"

            cve_list.append([cve_id, description, cvss_score, status])

        df = pd.DataFrame(cve_list, columns=["CVE_ID", "Description", "CVSS_Score", "Status"])
        df.to_csv("cve_data.csv", index=False)
        print("✅ CVE Data saved to CSV")


if __name__ == "__main__":
    save_cve_to_csv()
