from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from qdrant_client import QdrantClient, models
import os
import base64
from PIL import Image
# import pillow_heif # Reverted change
import io
import uuid
import requests
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the embedding model
try:
    model = SentenceTransformer('clip-ViT-B-32')
    print("✅ Successfully loaded CLIP embedding model")
except Exception as e:
    print(f"❌ Failed to load CLIP embedding model: {e}")
    model = None

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333/")
COLLECTION_NAME = "multimodal-search"

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Register HEIF plugin # Reverted change
# pillow_heif.register_heif_opener() # Reverted change

def initialize_collection():
    """Initialize the Qdrant collection if it doesn't exist"""
    if model is None:
        print("❌ Cannot initialize collection: Embedding model not loaded")
        return False
        
    if not client.collection_exists(COLLECTION_NAME):
        try:
            # Create sample embeddings to get vector size
            sample_text = "sample text"
            sample_text_embedding = model.encode(sample_text).tolist()
            
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "image": models.VectorParams(size=len(sample_text_embedding), distance=models.Distance.COSINE),
                    "text": models.VectorParams(size=len(sample_text_embedding), distance=models.Distance.COSINE),
                }
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize collection: {e}")
            return False
    else:
        print(f"✅ Collection {COLLECTION_NAME} already exists")
        return True

@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload an image with caption to the database"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        caption = request.form.get('caption', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # If no caption provided, use a generic one
        if not caption.strip():
            caption = f"Image: {file.filename}"
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(filepath)
        
        # Check if model is loaded
        if model is None:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': 'Embedding model not loaded. Please check server logs.'
            }), 500
        
        try:
            # Generate embeddings
            text_embedding = model.encode(caption).tolist()
            
            # For image embedding, process the actual image
            img = PILImage.open(filepath)
            image_embedding = model.encode(img).tolist()
        except Exception as e:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"❌ Error generating embeddings: {e}")
            return jsonify({
                'error': f'Failed to generate embeddings: {str(e)}'
            }), 500
        
        # Ensure collection exists before uploading
        if not client.collection_exists(COLLECTION_NAME):
            try:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        "image": models.VectorParams(size=len(text_embedding), distance=models.Distance.COSINE),
                        "text": models.VectorParams(size=len(text_embedding), distance=models.Distance.COSINE),
                    }
                )
                print(f"✅ Created collection: {COLLECTION_NAME}")
            except Exception as e:
                print(f"❌ Failed to create collection: {e}")
                # Clean up the uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'Failed to create collection: {str(e)}'}), 500
        
        # Upload to Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=file_id,
                    vector={
                        "text": text_embedding,
                        "image": image_embedding,
                    },
                    payload={
                        "caption": caption,
                        "image_path": filepath,
                        "filename": filename
                    }
                )
            ]
        )
        
        return jsonify({
            'message': 'Image uploaded successfully',
            'id': file_id,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for images using text query"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'text-to-image')  # 'text-to-image' or 'image-to-text'
        limit = data.get('limit', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Embedding model not loaded. Please check server logs.'
            }), 500
        
        try:
            # Generate query embedding
            query_embedding = model.encode(query).tolist()
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            return jsonify({
                'error': f'Failed to generate query embedding: {str(e)}'
            }), 500
        
        # Ensure collection exists before searching
        if not client.collection_exists(COLLECTION_NAME):
            return jsonify({
                'error': 'Collection does not exist. Please upload some images first.',
                'query': query,
                'results': [],
                'total': 0
            }), 404
        
        # Search in Qdrant
        if search_type == 'text-to-image':
            # Search for images using text query
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                using="image",
                with_payload=True,
                limit=limit
            )
        else:
            # Search for text using text query (for demonstration)
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                using="text",
                with_payload=True,
                limit=limit
            )
        
        # Format results
        results = []
        for point in search_result.points:
            # Convert score to distance (for cosine similarity: distance = 1 - score)
            distance = 1.0 - point.score
            result = {
                'id': point.id,
                'distance': distance,
                'score': point.score,  # Keep score for reference
                'caption': point.payload.get('caption', ''),
                'filename': point.payload.get('filename', ''),
                'image_url': f"/image/{point.payload.get('filename', '')}"
            }
            results.append(result)
        
        return jsonify({
            'query': query,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<filename>')
def serve_image(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/collection/info')
def collection_info():
    """Get information about the collection"""
    try:
        collection_exists = client.collection_exists(COLLECTION_NAME)
        if collection_exists:
            info = client.get_collection(COLLECTION_NAME)
            count_result = client.count(COLLECTION_NAME) # Renamed to avoid conflict with built-in count
            return jsonify({
                'exists': True,
                'count': count_result.count,
                'status': 'Online',
                'vectors_config': str(info.config.params.vectors) # Ensure this is serializable
            })
        else:
            return jsonify({
                'exists': False,
                'count': 0,
                'status': 'Offline or Not Initialized'
            })
    except Exception as e:
        print(f"❌ Error in /collection/info: {str(e)}")
        # Provide a more structured error response for the frontend
        return jsonify({
            'exists': False, 
            'count': 0, 
            'status': 'Error retrieving info', 
            'error': str(e)
        }), 500

@app.route('/delete_image/<file_id>', methods=['DELETE'])
def delete_image_route(file_id):
    """Delete an image from the filesystem and Qdrant"""
    try:
        if not file_id:
            return jsonify({'error': 'No file_id provided'}), 400

        # Try to retrieve the point by its ID
        try:
            points_to_retrieve = [file_id]
            # Corrected: client.get_points returns a list of PointStruct directly
            retrieved_points_list = client.get_points(
                collection_name=COLLECTION_NAME,
                ids=points_to_retrieve,
                with_payload=True # Ensure we get the payload
            )
        except Exception as e:
            print(f"❌ Error retrieving point {file_id} for deletion: {e}")
            return jsonify({'error': f'Error finding image details: {str(e)}'}), 500

        if not retrieved_points_list: # Check if the list is empty
            return jsonify({'error': 'Image not found in database'}), 404
        
        point_to_delete = retrieved_points_list[0] # Access the first element of the list
        image_filename = point_to_delete.payload.get('filename')

        if not image_filename:
            print(f"⚠️ Warning: file_id {file_id} found in Qdrant but missing 'filename' in payload.")
            try:
                client.delete_points(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=[file_id])
                )
                return jsonify({'message': 'Image metadata deleted from database (file not found on disk due to missing payload info)'}), 200
            except Exception as e:
                print(f"❌ Error deleting point {file_id} from Qdrant (filename was missing): {e}")
                return jsonify({'error': f'Failed to delete image metadata from database: {str(e)}'}), 500

        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        # 1. Delete from Qdrant
        try:
            client.delete_points(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=[file_id])
            )
            print(f"✅ Successfully deleted point {file_id} from Qdrant.")
        except Exception as e:
            print(f"❌ Failed to delete point {file_id} from Qdrant: {e}")
            return jsonify({'error': f'Failed to delete image from database: {str(e)}'}), 500

        # 2. Delete from filesystem
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"✅ Successfully deleted file {image_path} from filesystem.")
            except OSError as e:
                print(f"❌ Failed to delete file {image_path} from filesystem: {e}")
                return jsonify({
                    'message': 'Image deleted from database, but failed to delete file from disk. Please check server logs.',
                    'warning': str(e)
                }), 207 # Multi-Status or 500 (Consider 200 with a warning in payload too)
        else:
            print(f"⚠️ File {image_path} not found on filesystem for deletion (was already deleted or path is incorrect).")

        return jsonify({'message': 'Image deleted successfully', 'id_deleted': file_id}), 200

    except Exception as e:
        print(f"❌ Unexpected error in delete_image_route for {file_id}: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze what features the model extracts from an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporary file
        temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
        temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_filepath)
        
        try:
            # Check if model is loaded
            if model is None:
                return jsonify({
                    'error': 'Embedding model not loaded. Please check server logs.'
                }), 500
            
            try:
                # Get image embedding from the actual image
                img = PILImage.open(temp_filepath)
                image_embedding = model.encode(img).tolist()
            except Exception as e:
                print(f"❌ Error generating image embedding: {e}")
                return jsonify({
                    'error': f'Failed to generate image embedding: {str(e)}'
                }), 500
            
            # Find similar images in the database
            if client.collection_exists(COLLECTION_NAME):
                search_result = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=image_embedding,
                    using="image",
                    with_payload=True,
                    limit=5
                )
                
                similar_images = []
                for point in search_result.points:
                    # Convert score to distance (for cosine similarity: distance = 1 - score)
                    distance = 1.0 - point.score
                    similar_images.append({
                        'caption': point.payload.get('caption', ''),
                        'filename': point.payload.get('filename', ''),
                        'distance': distance,
                        'similarity': point.score  # Keep score for reference
                    })
            else:
                similar_images = []
            
            return jsonify({
                'message': 'Image analyzed successfully',
                'embedding_size': len(image_embedding),
                'similar_images': similar_images,
                'analysis': 'The model extracted visual features including objects, colors, composition, and scene elements'
            })
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Multimodal search API is running'})

if __name__ == '__main__':
    if model is None:
        print("❌ Failed to load embedding model. Server will not start.")
        print("Please check your dependencies and model installation.")
    else:
        print("🚀 Initializing collection...")
        if initialize_collection():
            print("✅ Collection initialized successfully")
            print("🌐 Starting Flask server on http://localhost:5001...")
            app.run(debug=True, host='0.0.0.0', port=5001)
        else:
            print("❌ Failed to initialize collection. Server will not start.") 