// Jenkins version of the CI pipeline. Needs an agent with Python 3.12 and Docker,
// plus a "ghcr-creds" username/password credential for pushing images.
pipeline {
  agent any
  environment { IMAGE = "ghcr.io/palakgupta7456/crypto-anomaly-detector" }
  options { timestamps(); timeout(time: 40, unit: 'MINUTES') }
  stages {
    stage('Setup') {
      steps {
        sh '''
          python3 -m venv .venv && . .venv/bin/activate
          pip install -q torch --index-url https://download.pytorch.org/whl/cpu
          pip install -q -e ".[train,api,dev]" onnxscript
        '''
      }
    }
    stage('Lint') { steps { sh '. .venv/bin/activate && ruff check .' } }
    stage('Train') { steps { sh '. .venv/bin/activate && python -m cad.synthetic && python -m cad.train' } }
    stage('Test') {
      steps { sh '. .venv/bin/activate && pytest --junitxml=reports/junit.xml' }
      post { always { junit 'reports/junit.xml' } }
    }
    stage('Docker build') { steps { sh 'docker build -t $IMAGE:$GIT_COMMIT .' } }
    stage('Push') {
      when { branch 'main' }
      steps {
        withCredentials([usernamePassword(credentialsId: 'ghcr-creds', usernameVariable: 'U', passwordVariable: 'P')]) {
          sh 'echo $P | docker login ghcr.io -u $U --password-stdin && docker push $IMAGE:$GIT_COMMIT'
        }
      }
    }
  }
  post { always { cleanWs() } }
}
