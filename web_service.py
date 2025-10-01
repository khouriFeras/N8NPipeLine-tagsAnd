from flask import Flask, request, jsonify, send_file
import os
from pathlib import Path
import tempfile
import uuid
from apis.universal_translation_seo import process_excel_file
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
app = Flask(__name__)
# Store generated files temporarily
generated_files = {}
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "translation-web-service"})
    
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Translation web service is running!",
        "endpoints": {
            "health": "GET /health",
            "test": "GET /test", 
            "process": "POST /process",
            "translate": "POST /translate",
            "download": "GET /download/<filename>"
        }
    })

@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        csv_content = data.get('csv_content', '')
        product_context = data.get('product_context', 'tools and equipment')
        sample = int(data.get('sample', 5))
        
        if not csv_content:
            return jsonify({"error": "No CSV content provided"}), 400
            
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_csv:
            temp_csv.write(csv_content)
            temp_csv_path = temp_csv.name
            
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        try:
            # Process the file (run async function)
            import asyncio
            result = asyncio.run(process_excel_file(
                input_path=Path(temp_csv_path),
                output_path=Path(temp_output_path),
                name_col="Title",
                desc_col="Body (HTML)",
                product_context=product_context,
                sample=sample,
                api_key=os.getenv("OPENAI_API_KEY", "")
            ))
            
            output_size = os.path.getsize(temp_output_path) if os.path.exists(temp_output_path) else 0
            file_id = str(uuid.uuid4())

            generated_files[file_id] = temp_output_path
            
            return jsonify({
                "success": True,
                "message": "Translation and SEO processing completed successfully",
                "output_size": output_size,
                "result": result,
                "sample_processed": sample,
                "file_id": file_id,  # Return the file ID instead of full path
                "output_columns": [
                    "arabic description", "product_title_ar", "meta_title", "meta_description", "handle_ar"
                ],
                "features": {
                    "translation": "Arabic product descriptions and titles",
                    "seo": "Meta titles, descriptions, and SEO-friendly handles",
                    "combined": "All data appended to new columns in the output file"
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
            
        finally:
            # Clean up temp CSV file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Same logic as /process but with different endpoint name
        return process_data()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    try:
        # Look up the file path using the file_id
        if file_id not in generated_files:
            return jsonify({"error": f"File ID not found: {file_id}"}), 404
            
        file_path = generated_files[file_id]
        
        # Check if file exists
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name='translated_products.xlsx',
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            return jsonify({"error": f"File not found: {file_path}"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (for deployment platforms)
    port = int(os.environ.get('PORT', 5000))
    # Run in production mode when deployed
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)