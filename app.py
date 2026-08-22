from flask import Flask, render_template, request, session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import requests
from model.predict import predict_disease

app = Flask(__name__)
app.secret_key = "agrivision_secret_key"

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "static",
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= HOME =================

@app.route("/")
def home():
    return render_template("index.html")


# ================= REGISTER =================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        full_name = request.form["full_name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]

        # Hash the password
        password_hash = generate_password_hash(
    password,
    method="pbkdf2:sha256"
)

        conn = sqlite3.connect("agrivision.db", timeout=10)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO farmers(full_name, email, phone, password)
            VALUES (?, ?, ?, ?)
        """, (full_name, email, phone, password_hash))

        conn.commit()
        conn.close()

        return "<h2>🎉 Registration Successful!</h2>"

    return render_template("register.html")


# ================= LOGIN =================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("agrivision.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM farmers WHERE email=?",
            (email,)
        )

        farmer = cursor.fetchone()

        conn.close()

        if farmer and check_password_hash(farmer[4], password):

            session["farmer_id"] = farmer[0]

            return render_template("dashboard.html")

        else:
            return "❌ Invalid Email or Password"

    return render_template("login.html")

# ================= UPLOAD & AI PREDICTION =================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    if request.method == "POST":

        image = request.files["leaf"]

        if image.filename != "":

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                image.filename
            )

            image.save(filepath)

            # AI prediction
            disease, confidence = predict_disease(filepath)

            # Save prediction to database
            conn = sqlite3.connect("agrivision.db")
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions
                (image_name, disease, confidence)
                VALUES (?, ?, ?)
            """, (image.filename, disease, confidence))

            conn.commit()
            conn.close()

            return render_template(
                "result.html",
                disease=disease,
                confidence=round(confidence, 2),
                image=image.filename
            )

    return render_template("upload.html")


# ================= DASHBOARD =================

# ================= DASHBOARD =================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("agrivision.db")
    cursor = conn.cursor()

    # Total predictions
    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total_predictions = cursor.fetchone()[0]

    # Healthy plants
    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE LOWER(disease) LIKE '%healthy%'
    """)

    healthy_plants = cursor.fetchone()[0]

    # Diseased plants
    diseased_plants = total_predictions - healthy_plants

    # Average confidence
    cursor.execute("""
        SELECT AVG(confidence)
        FROM predictions
    """)

    average_confidence = cursor.fetchone()[0]

    # Disease prediction counts
    cursor.execute("""
        SELECT disease, COUNT(*)
        FROM predictions
        GROUP BY disease
        ORDER BY COUNT(*) DESC
    """)

    disease_counts = cursor.fetchall()

    conn.close()

    if average_confidence is None:
        average_confidence = 0

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        healthy_plants=healthy_plants,
        diseased_plants=diseased_plants,
        average_confidence=round(average_confidence, 2),
        disease_counts=disease_counts
    )
# ================= WEATHER DASHBOARD =================

# ================= WEATHER DASHBOARD =================

# ================= WEATHER DASHBOARD =================

@app.route("/weather")
def weather():

    if "farmer_id" not in session:
        return redirect("/login")

    # Bengaluru coordinates
    latitude = 12.9716
    longitude = 77.5946

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current=temperature_2m,relative_humidity_2m,"
        "wind_speed_10m,weather_code"
        "&daily=precipitation_probability_max"
        "&timezone=auto"
    )

    try:

        response = requests.get(url, timeout=10)
        weather_data = response.json()

        current = weather_data["current"]

        temperature = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        weather_code = current["weather_code"]

        rainfall_probability = (
            weather_data["daily"]["precipitation_probability_max"][0]
        )

        # Convert weather code to readable condition
        if weather_code == 0:
            condition = "Clear Sky ☀️"

        elif weather_code in [1, 2, 3]:
            condition = "Partly Cloudy ☁️"

        elif weather_code in [45, 48]:
            condition = "Foggy 🌫️"

        elif weather_code in [51, 53, 55, 56, 57]:
            condition = "Drizzle 🌦️"

        elif weather_code in [61, 63, 65, 66, 67]:
            condition = "Rain 🌧️"

        elif weather_code in [71, 73, 75, 77]:
            condition = "Snow ❄️"

        elif weather_code in [80, 81, 82]:
            condition = "Rain Showers 🌧️"

        elif weather_code in [95, 96, 99]:
            condition = "Thunderstorm ⛈️"

        else:
            condition = "Unknown Weather"

    except Exception as e:

        print("Weather API error:", e)

        temperature = "--"
        humidity = "--"
        wind_speed = "--"
        rainfall_probability = "--"
        condition = "Weather unavailable ⚠️"

    return render_template(
        "weather.html",
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        rainfall_probability=rainfall_probability,
        condition=condition
    )
# ================= AI CHAT ASSISTANT =================

@app.route("/chat", methods=["GET", "POST"])
def chat():

    if "farmer_id" not in session:
        return redirect("/login")

    if request.method == "GET":
        return render_template("chat.html")

    data = request.get_json()

    user_message = data.get("message", "").strip().lower()

    if not user_message:
        return {"reply": "Please enter a question. 🌱"}

    # Simple AgroVision farming assistant
    if "tomato" in user_message and "disease" in user_message:
        reply = (
            "🍅 Common tomato diseases include early blight, late blight, "
            "bacterial spot and leaf mold. Remove infected leaves, maintain "
            "good air circulation and avoid unnecessary overhead watering."
        )

    elif "late blight" in user_message:
        reply = (
            "🦠 For late blight, remove infected plant material promptly, "
            "avoid wetting the leaves and maintain good air circulation. "
            "For serious infections, consult a local agricultural expert "
            "about an appropriate fungicide."
        )

    elif "water" in user_message or "irrigation" in user_message:
        reply = (
            "💧 Check the soil moisture before irrigation. Avoid excessive "
            "watering and avoid keeping the leaves continuously wet."
        )

    elif "healthy" in user_message or "health" in user_message:
        reply = (
            "🌱 To keep crops healthy, use healthy planting material, "
            "provide appropriate water and nutrition, maintain good "
            "air circulation and inspect plants regularly."
        )

    elif "weather" in user_message or "rain" in user_message:
        reply = (
            "🌦️ You can check the AgroVision Weather Dashboard for current "
            "temperature, humidity, wind speed and rainfall probability."
        )

    elif "pepper" in user_message:
        reply = (
            "🌶️ For healthy pepper plants, use disease-free planting "
            "material, avoid excessive moisture on leaves and inspect "
            "plants regularly for bacterial or fungal symptoms."
        )

    elif "potato" in user_message:
        reply = (
            "🥔 Monitor potato plants regularly for early blight and late "
            "blight. Remove infected plant material and maintain good "
            "field sanitation."
        )

    elif "hello" in user_message or "hi" in user_message:
        reply = (
            "👋 Hello! I'm the AgroVision AI Assistant. "
            "Ask me about crop diseases, prevention, irrigation or weather."
        )

    else:
        reply = (
            "🤖 I can help with crop diseases, prevention, irrigation, "
            "weather and general crop-care guidance. Please ask me a "
            "specific farming question."
        )

    return {"reply": reply}

# ================= PREDICTION HISTORY =================

@app.route("/history")
def history():

    search = request.args.get("search", "").strip()

    conn = sqlite3.connect("agrivision.db")
    cursor = conn.cursor()

    if search:
        cursor.execute("""
            SELECT id, image_name, disease, confidence, prediction_date
            FROM predictions
            WHERE disease LIKE ?
            ORDER BY id DESC
        """, ("%" + search + "%",))
    else:
        cursor.execute("""
            SELECT id, image_name, disease, confidence, prediction_date
            FROM predictions
            ORDER BY id DESC
        """)

    predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions,
        search=search
    )

# ================= DELETE PREDICTION =================

@app.route("/delete_prediction/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):

    # Check whether farmer is logged in
    if "farmer_id" not in session:
        return "❌ Please login first."

    conn = sqlite3.connect("agrivision.db")
    cursor = conn.cursor()

    # Delete the selected prediction
    cursor.execute(
        "DELETE FROM predictions WHERE id = ?",
        (prediction_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/history")




# ================= FARMER PROFILE =================

@app.route("/profile")
def profile():

    # Check whether farmer is logged in
    if "farmer_id" not in session:
        return "❌ Please login first."

    farmer_id = session["farmer_id"]

    conn = sqlite3.connect("agrivision.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT full_name, email, phone
        FROM farmers
        WHERE id = ?
    """, (farmer_id,))

    farmer = cursor.fetchone()

    conn.close()

    return render_template(
        "profile.html",
        farmer=farmer
    )


# ================= LOGOUT =================

@app.route("/logout")
def logout():

    session.clear()

    return render_template("index.html")


# ================= RUN APPLICATION =================

if __name__ == "__main__":
    app.run(debug=True)