"""Small Qwen VLM behind a loopback OpenAI-compatible chat API."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--port', type=int, default=18881)
    p.add_argument('--model-name', default='qwen-small')
    args = p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True,
                                             min_pixels=256*28*28, max_pixels=1280*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16, device_map='cuda:0', local_files_only=True,
        attn_implementation='sdpa').eval()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200,dict(status='ready',model=args.model_name))
            elif self.path == '/v1/models':
                self.reply(200,dict(object='list',data=[dict(id=args.model_name,object='model')]))
            else:
                self.reply(404,dict(error='Unknown route'))

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.reply(404,dict(error='Unknown route'))
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 24*1024*1024:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if body.get('stream') or body.get('model') != args.model_name:
                    raise ValueError('Unsupported stream or model')
                messages, images = [], []
                for message in body['messages']:
                    content = message['content']
                    if isinstance(content,str):
                        messages.append(dict(role=message['role'],content=content))
                        continue
                    pieces = []
                    for item in content:
                        if item['type'] == 'text':
                            pieces.append(dict(type='text',text=item['text']))
                        elif item['type'] == 'image_url':
                            url = item['image_url']['url']
                            if not url.startswith('data:image/') or ';base64,' not in url:
                                raise ValueError('Only embedded images are supported')
                            im = Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1],validate=True))).convert('RGB')
                            images.append(im)
                            pieces.append(dict(type='image',image=im))
                        else:
                            raise ValueError('Unsupported message content')
                    messages.append(dict(role=message['role'],content=pieces))
                maximum = int(body.get('max_tokens',2048))
                if not 1 <= maximum <= 12288:
                    raise ValueError('Invalid max_tokens')
                text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                with lock, torch.inference_mode():
                    inputs = processor(text=[text],images=images or None,padding=True,return_tensors='pt').to(model.device)
                    generated = model.generate(**inputs,max_new_tokens=maximum,do_sample=False)
                    result = generated[:,inputs.input_ids.shape[1]:]
                    answer = processor.batch_decode(result,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
                self.reply(200,dict(id='local-'+str(time.time_ns()),object='chat.completion',
                    model=args.model_name,choices=[dict(index=0,message=dict(role='assistant',content=answer),
                    finish_reason='length' if result.shape[1]==maximum else 'stop')],
                    usage=dict(prompt_tokens=inputs.input_ids.shape[1],completion_tokens=result.shape[1],total_tokens=generated.shape[1])))
            except ValueError as error:
                self.reply(400,dict(error=str(error)))
            except Exception as error:
                self.reply(500,dict(error=type(error).__name__))

    print('Qwen API ready on loopback port', args.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__':
    main()
