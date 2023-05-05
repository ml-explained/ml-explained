# executes all .py files located in the scripts directory used to generate media content for the website

# loop through all "*.py" files in the scripts directory and swap "/" with "."
for file in `find scripts -name '*.py' | tr / .`
do  
    # remove the ".py" ending
    script=${file//.py/}

    # echo file attempting to execute
    echo -n $file

    # if script executed successfully (ensure you have a venv "env" with required dependencies listed in requirements.txt)
    if `env/bin/python -m $script > /dev/null 2>&1` # silently run the python scripts (no error messages)
    then
        # tick
        echo -e ' \u2714'
    else
        # red cross when script did not successfully complete
        echo -e ' \u274c'
    fi
done
