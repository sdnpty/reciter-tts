#!/bin/bash
# Скрипт для скачивания ONNX Runtime Android AAR из Maven Central
# Использование: ./download_onnxruntime.sh [VERSION]
# По умолчанию: 1.26.0

VERSION="${1:-1.26.0}"
AAR_FILE="onnxruntime-android-${VERSION}.aar"
URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${VERSION}/${AAR_FILE}"
OUT_DIR="app/libs"

echo "Downloading ONNX Runtime Android ${VERSION}..."
echo "URL: ${URL}"

mkdir -p "${OUT_DIR}"

if command -v wget &> /dev/null; then
    wget -O "${OUT_DIR}/${AAR_FILE}" "${URL}"
elif command -v curl &> /dev/null; then
    curl -L -o "${OUT_DIR}/${AAR_FILE}" "${URL}"
else
    echo "Error: wget or curl required"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "Success: ${OUT_DIR}/${AAR_FILE}"
    echo "Size: $(du -h ${OUT_DIR}/${AAR_FILE} | cut -f1)"
else
    echo "Download failed"
    exit 1
fi
