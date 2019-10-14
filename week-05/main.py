from flask import Flask, render_template, url_for
import paypalrestsdk
import logging

app = Flask(__name__)

paypalrestsdk.configure({
  "mode": "sandbox", # sandbox or live
  "client_id": "Ad7MPkB8s5FTc41QoeRS3QXcl5HCXNhK-vZtB3FFm88Sj4p80MZfIJUOX0qnjGMA4Qh4oPK_jLQPkY2j",
  "client_secret": "EKaZvBNbJTeCYWm5EumZ4O0Faw35a4ea7Dg__PyVtOyBdZuNwsliWvO0bkdKaGoDQXLxo6E0Quf7RPPq" })

@app.route("/")
@app.route("/login")
def home():
    return render_template('index.html', title="Safe AI - Login")

@app.route("/home")
@app.route("/detection")
def about():
    return render_template('detection.html', title="Safe AI - Detection")

@app.route("/forgot-password")
def forgotPass():
    return render_template('forgot-password.html', title="Safe AI - Forgot Password")

@app.route("/register")
def register():
    return render_template('register.html', title="Safe AI - Register")

@app.route("/pay", methods = ['POST'])
def pay():
    print('Chegou /pay')
    payment = paypalrestsdk.Payment({
        "intent": "sale",
        "payer": {
            "payment_method": "paypal"},
        "redirect_urls": {
            "return_url": "http://localhost:3000/success",
            "cancel_url": "http://localhost:3000/cancel"},
        "transactions": [{
            "item_list": {
                "items": [{
                    "name": "Service Instance",
                    "sku": "001",
                    "price": "5.00",
                    "currency": "USD",
                    "quantity": 1}]},
            "amount": {
                "total": "5.00",
                "currency": "USD"},
            "description": "This is the payment transaction description."}]})

    if payment.create():
        print("Payment created successfully")
        for link in payment.links:
            if link.rel == "approval_url":
                # Convert to str to avoid Google App Engine Unicode issue
                # https://github.com/paypal/rest-api-sdk-python/pull/58
                approval_url = str(link.href)
                print("Redirect for approval: %s" % (approval_url))
    else:
        print(payment.error)

    

if __name__ == '__main__':
    app.run(debug=True)

