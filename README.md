# mnist-web

Web app allowing the user to draw digits and automatically analyzes them using an ANN model.

### Install
This app uses [pipenv](https://github.com/pypa/pipenv) for handling virtual environments and dependencies. It can be installed via pip `pip install pipenv`. Once all dependencies are in place, the project is run in development mode by using
```bash
export DEBUG=True; python run.py
```

### Deployment
This app uses container-based Heroku deployment, but should only need minimal tweaking to be adapted to other Cloud Foundry services.
