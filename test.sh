curl -X POST "http://localhost:8080/long_synthesize" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "nói theo mình nhé, bear, bạn nói chậm rõ ràng nhé. xin chào mọi người nhé. tên mình là tú anh, mình viết câu này rất dài để test xem có hoạt động ổn định không nhé <chuckle>",
           "speaker": "female_happy"
         }' \
     --output output_eng.wav
