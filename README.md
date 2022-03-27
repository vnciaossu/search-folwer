Using: python 3.10.4

Package need setting:
pip install opencv-python

python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

Using:

1. Resize tất cả bức ảnh trong thư mục train về cùng 1 kích thước là 600x600 bằng file Resize_data.py
2. Chạy File Create_database.py để tạo thư viện thuộc tính trích rút đặc trưng bức ảnh, gắn nhãn cho bức ảnh.
   Line 48 là đường dẫn đến thư mục datatrain
3. Chạy Search_Flower.py để nhận diện bức ảnh đầu vào (Line 63) với bộ dữ liệu trích rút đặc trưng ở Line 54
