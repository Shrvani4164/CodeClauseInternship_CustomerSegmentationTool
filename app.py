from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/cluster", methods=["POST"])
def cluster():
    try:
        age = float(request.form["age"])
        income = float(request.form["income"])
        spending_score = float(request.form["spending_score"])
        social_media = request.form["social_media"]
        streaming_hours = float(request.form["streaming_hours"])
        gaming_hours = float(request.form["gaming_hours"])

        data = pd.DataFrame({
            "Age": [18, 22, 28, 35, 40],
            "Income": [15000, 30000, 45000, 60000, 75000],
            "Spending Score": [80, 70, 60, 50, 40],
            "Social Media": ["Instagram", "TikTok", "Snapchat", "Instagram", "TikTok"],
            "Streaming Hours": [15, 10, 20, 25, 30],
            "Gaming Hours": [10, 5, 15, 20, 8]
        })

        new_data = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "Spending Score": spending_score,
            "Social Media": social_media,
            "Streaming Hours": streaming_hours,
            "Gaming Hours": gaming_hours
        }])
        data = pd.concat([data, new_data], ignore_index=True)

        transformer = ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(), ["Social Media"])],
            remainder="passthrough"
        )
        
        scaler = StandardScaler()
        data_scaled = transformer.fit_transform(data)
        data_scaled = scaler.fit_transform(data_scaled)

        kmeans = KMeans(n_clusters=5, random_state=42, init='random')
        data["Cluster"] = kmeans.fit_predict(data_scaled)
        user_cluster = data.iloc[-1]["Cluster"]
        return f"You belong to Cluster: {int(user_cluster)}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
