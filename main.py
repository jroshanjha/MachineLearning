from flask import Flask
  
app = Flask(__name__)
  
# Pass the required route to the decorator.
@app.route("/home-page")
def home_page():
    return "<p><h1>this is home page for client side handller!</h1></p>"

@app.route("/about-page")
def about_page():
    return "<div style='color:red;background-color:yellow;'> this is about us page for admin!!!</div>"
 
@app.route("/")
def index():
    return "<div><h1>this is machine learning first class by me!!!!</h1></div>"
if __name__ == "__main__":
    app.run(debug=True)


# https://alpha.bito.ai/auth/login?red=ws&A_vRP5kyFV48qNc7G2zy_uqR1SvVFabBp2kzvFUqk80=