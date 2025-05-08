# Setting up the environment

1. Install miniconda
2. Create an environment

    ```shell
    conda create -n mastermind python=3
    ```
3. Activate the environment (in case you need to use it in the terminal)

    ```shell
    conda activate mastermind
    ```
4. Install required packages
    ```shell
    pip install -r requirements.txt
    ```
5. In case you need to use it in Visual Studio Code (VSCode) select the created environment.

# Converting mp4 to m4a

## On macOS
```shell
afconvert {filename}.mp4 -o {directory}/{filename}.m4a  -f m4af -d aac
```