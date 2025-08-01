from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '689339c29e747c6ca95621fb402bc22adfb1c09ec43505dab043b5c03f72f23f'

# Import routes at the bottom to avoid circular imports
from app import routes