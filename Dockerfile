# Sử dụng base image python slim để tối ưu dung lượng
FROM python:3.11-slim-bookworm

# 1. Cài đặt uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file định nghĩa dependency
COPY pyproject.toml uv.lock ./

# 4. Cài đặt Python dependencies
RUN uv sync --frozen --no-cache

# 5. Fetch Camoufox Browser (Quan trọng)
# Tải binary browser ngay lúc build để không phải tải mỗi lần chạy container
RUN uv run python -m camoufox fetch

# 6. Copy source code và file cấu hình
COPY src ./src
COPY .env .
# Copy models.json theo yêu cầu
COPY models.json .

# Expose port
EXPOSE 8000

# Chạy app
CMD ["uv", "run", "python", "src/main.py"]
