from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore, auth
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import os 
import json
from google.oauth2 import service_account #ini baru nanti dihapus ya
from google.cloud import firestore #ini baru nanti dihapus ya
from google.auth import exceptions #ini baru nanti dihapus ya

# Initialize Flask application
app = Flask(__name__)   

# Mengambil credential
#KEY_JSON = os.environ.get('KEY_JSON')

# Konfigurasi credential
#app.config['KEY_JSON'] = KEY_JSON

#cred = credentials.Certificate(app.config['KEY_JSON'])  # Replace with your own service account key file path

# Initialize Firestore
#inisebelum_firebase_admin.initialize_app(cred)
#firebase_admin.initialize_app() #nanti yg ini dihapus
db = firestore.Client() #ini diganti c kecil dihapus ya

#read key.json
#with open('key.json', 'r') as file:
	#key_json = file.read()

#load json content
#cred_json = json.loads(key_json)

#use cred_json
#cred = credentials.Certificate(cred_json)


@app.route('/', methods=['GET'])
def fundup():
    return  "server success!"

@app.route('/startup', methods=['POST'])
def addRecStartup():
    startup_features_ref = db.collection('startup')
    investor_features_ref = db.collection('investor_loker')

    startup_features_docs = startup_features_ref.stream()        
    startup_features = []
    startup_ids = []
    for doc in startup_features_docs:
        data = doc.to_dict()
        startup_features.append(data['tingkat_perkembangan_perusahaan'] + ' ' + data['industri_startup'])
        startup_ids.append(str(doc.id))
        
    investor_features_docs = investor_features_ref.stream()            
    investor_features = []
    investor_ids = []
    for doc in investor_features_docs:
        data = doc.to_dict()
        investor_features.append(data['target_perkembangan'] + ' ' + data['target_industri'])
        investor_ids.append(doc.id)            
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(startup_features + investor_features)
    startup_sequences = tokenizer.texts_to_sequences(startup_features)
    startup_padded = pad_sequences(startup_sequences)
        
    investor_sequences = tokenizer.texts_to_sequences(investor_features)
    investor_padded = pad_sequences(investor_sequences)
        
        # Convert padded sequences to tensors
    startup_tensors = tf.convert_to_tensor(startup_padded, dtype=tf.float32)
    investor_tensors = tf.convert_to_tensor(investor_padded, dtype=tf.float32)
        
        # Calculate cosine similarity between startup and investor tensors
    similarity_matrix = cosine_similarity(startup_tensors, investor_tensors)
        
    def get_investor_matches(startup_id):
        matches = {}
        startup_index = startup_ids.index(startup_id)
        similarities = similarity_matrix[startup_index]
        sorted_indexes = np.argsort(similarities)[::-1]
        top_matches = [investor_ids[i] for i in sorted_indexes[:20]]
        matches[startup_id] = top_matches
        return matches
        
        # Add investor matches to Firestore collection
    def add_investor_matches(startup_id, investor_matches):
        matches_ref = db.collection('investor_matches')
        matches_ref.document(startup_id).set({ 'investor_matches': investor_matches })
        
    for id in startup_ids:
        input_id = id
        investor_matches = get_investor_matches(input_id)
        add_investor_matches(input_id, investor_matches[input_id])
    return jsonify({'message': 'Startup data added successfully'})


@app.route('/investor', methods=['POST'])
def addRecInvestor(): 
    investor_features_ref = db.collection('investor_loker')
    startup_features_ref = db.collection('startup')
    investor_features_docs = investor_features_ref.stream()
        
    investor_features = []
    investor_ids = []
    for doc in investor_features_docs:
        data = doc.to_dict()
        investor_features.append(data['target_perkembangan'] + ' ' + data['target_industri'])
        investor_ids.append(str(doc.id))
        
    startup_features_docs = startup_features_ref.stream()            
    startup_features = []
    startup_ids = []
    for doc in startup_features_docs:
        data = doc.to_dict()
        startup_features.append(data['tingkat_perkembangan_perusahaan'] + ' ' + data['industri_startup'])
        startup_ids.append(doc.id)      

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(investor_features + startup_features)
    investor_sequences = tokenizer.texts_to_sequences(investor_features)
    investor_padded = pad_sequences(investor_sequences)
        
    startup_sequences = tokenizer.texts_to_sequences(startup_features)
    startup_padded = pad_sequences(startup_sequences)
        
    # Convert padded sequences to tensors
    investor_tensors = tf.convert_to_tensor(investor_padded, dtype=tf.float32)
    startup_tensors = tf.convert_to_tensor(startup_padded, dtype=tf.float32)
        
    # Calculate cosine similarity between investor and startup tensors
    similarity_matrix = cosine_similarity(investor_tensors, startup_tensors)
        
    def get_startup_matches(investor_id):
        matches = {}
        investor_index = investor_ids.index(investor_id)
        similarities = similarity_matrix[investor_index]
        sorted_indexes = np.argsort(similarities)[::-1]
        top_matches = [startup_ids[i] for i in sorted_indexes[:20]]
        matches[investor_id] = top_matches
        return matches
        
        # Add investor matches to Firestore collection
    def add_startup_matches(investor_id, startup_matches):
        matches_ref = db.collection('startup_matches')
        matches_ref.document(investor_id).set({ 'startup_matches': startup_matches })
        
    for id in investor_ids:
        input_id = id
        startup_matches = get_startup_matches(input_id)
        add_startup_matches(input_id, startup_matches[input_id])
        
    return jsonify({'message': 'Investor data added successfully'})
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

@app.route('/get-recommendation', methods=['GET'])
def get_recomendation_for_startup():
    # Get the id_token from the request (assuming it's provided in the request)
    id_token = request.json.get('id_token')

    # Validate and authenticate the id_token (add your authentication logic here)

    # Check if the id_token is valid
    if id_token is None:
        return jsonify({'error': 'Invalid id_token'})

    # Convert id_token to input_id
    input_id = id_token

    # Query investor_matches collection to get the data
    query_ad = db.collection('investor_matches').document(input_id).get()

    # Check if the document exists
    if query_ad.exists:
        data = query_ad.to_dict()
        investor_matches = data.get('investor_matches', [])
        investor_ids_matches = []

        for investor_id in investor_matches:
            investor_ids_matches.append(investor_id)

        investor_loker_data = []

        for investor_id in investor_ids_matches:
            query = db.collection('investor_loker').document(investor_id).get()
            if query.exists:
                data = query.to_dict()
                investor_loker_data.append(data)
            else:
                print(f"No data found for investor ID: {investor_id}")

        # Process the retrieved investor_loker_data array as needed
        result = []
        for data in investor_loker_data:
            nama_lengkap = data['nama_lengkap']
            nik_investor = str(data['nik_investor'])
            email_investor = data['email_investor']
            target_industri = data['target_industri']
            target_perkembangan = data['target_perkembangan']
            # Add the processed data to the result list
            result.append({
                'nama_lengkap': nama_lengkap,
                'nik_investor': nik_investor,
                'email_investor': email_investor,
                'target_industri': target_industri,
                'target_perkembangan': target_perkembangan
            })

        return jsonify(result)
    else:
        query_ad = db.collection('startup_matches').document(input_id).get()
        data = query_ad.to_dict()
        startup_matches = data.get('startup_matches', [])
        startup_ids_matches = []
        
        for startup_id in startup_matches:
            startup_ids_matches.append(startup_id)
        
        startup_data = []

        for startup_id in startup_ids_matches:
            query = db.collection('startup').document(startup_id).get()
            if query.exists:
                data = query.to_dict()
                startup_data.append(data)
            else:
                print(f"No data found for Startup ID: {startup_id}")

            # Process the retrieved investor_loker_data array as needed
        result = []
        for data in startup_data:
            nama_lengkap = data['nama_lengkap']
            nik_startup = str(data['nik_startup'])
            email_startup = data['email_startup']
            industri_startup = data['industri_startup']
            tingkat_perkembangan_perusahaan = data['tingkat_perkembangan_perusahaan']

            #lanjutin sesuai apa aja atribut yg mau ditampilin di homepage
            result.append({
                'nama_lengkap': nama_lengkap,
                'nik_startup': nik_startup,
                'email_startup': email_startup,
                'industri_startup': industri_startup,
                'tingkat_perkembangan_perusahaan': tingkat_perkembangan_perusahaan
            })

        return jsonify(result)
   


if __name__ == '__main__':
    app.run(port=5000)
