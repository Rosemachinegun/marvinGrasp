from setuptools import setup, find_packages


setup(
    name="dm-lingkong-grip-sdk",
    version="1.0.0",
    author="xx.xu",
    author_email="xx.xu@email.com",
    description="Python SDK for Lingkong electric gripper",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "grpcio>=1.75.1",
        "protobuf>=6.32.1",
    ],
    # 如果有额外数据文件（如 proto 文件），可以用 package_data
    # package_data={"lingkong_grip_sdk": ["*.proto"]},
)