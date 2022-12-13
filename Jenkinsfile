pipeline {
    agent any
    environment {
        DOCKER_HUB_REPO = "lintoai/linto-platform-diarization"
        DOCKER_HUB_CRED = 'docker-hub-credentials'

        VERSION = ''
    }

    stages{
        stage('Docker build for master branch'){
            when{
                branch 'pyannote'
            }
            steps {
                echo 'Publishing latest pyannote'
                script {
                    image = docker.build(env.DOCKER_HUB_REPO)
                    VERSION = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()

                    docker.withRegistry('https://registry.hub.docker.com', env.DOCKER_HUB_CRED) {
                        image.push("${VERSION}")
                        image.push('pyannote-latest')
                    }
                }
            }
        }

    }// end stages
}