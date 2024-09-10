def buildDockerfile(main_folder, dockerfilePath, image_name, version, changedFiles) {
    if (changedFiles.contains(main_folder) || changedFiles.contains('celery_app') || changedFiles.contains('http_server') || changedFiles.contains('document') || changedFiles.contains('docker-entrypoint.sh') || changedFiles.contains('healthcheck.sh') || changedFiles.contains('wait-for-it.sh')) {
        echo "Building Dockerfile for ${image_name} with version ${version} (using ${dockerfilePath})"

        script {
            def image = docker.build(image_name, "-f ${dockerfilePath} .")

            docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                if (version  == 'latest-unstable') {
                    image.push('latest-unstable')
                } else {
                    image.push('latest')
                    image.push(version)
                }
            }
        }
    }
}

pipeline {
    agent any
    environment {
        // DOCKER_HUB_REPO_PYBK   = "lintoai/linto-diarization-pybk" // DEPRECATED
        DOCKER_HUB_REPO_PYANNOTE = "lintoai/linto-diarization-pyannote"
        DOCKER_HUB_REPO_SIMPLE = "lintoai/linto-diarization-simple"
    }
    
    stages {
        stage('Docker build for master branch') {
            when {
                branch 'master'
            }
            steps {
                echo 'Publishing latest'
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    echo "My changed files: ${changedFiles}"

                    // // DEPRECATED
                    // version = sh(
                    //     returnStdout: true, 
                    //     script: "awk -v RS='' '/#/ {print; exit}' pybk/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    // ).trim()
                    // buildDockerfile('pybk', 'pybk/Dockerfile', env.DOCKER_HUB_REPO_PYBK, version, changedFiles)

                    version = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' simple/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()
                    buildDockerfile('simple', 'simple/Dockerfile', env.DOCKER_HUB_REPO_SIMPLE, version, changedFiles)

                    version = sh(
                        returnStdout: true, 
                        script: "awk -v RS='' '/#/ {print; exit}' pyannote/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()
                    buildDockerfile('pyannote', 'pyannote/Dockerfile', env.DOCKER_HUB_REPO_PYANNOTE, version, changedFiles)
                }
            }
        }

        stage('Docker build for next (unstable) branch') {
            when {
                branch 'next'
            }
            steps {
                echo 'Publishing unstable'
                script {
                    def changedFiles = sh(returnStdout: true, script: 'git diff --name-only HEAD^ HEAD').trim()
                    echo "My changed files: ${changedFiles}"
                    
                    version = 'latest-unstable'

                    // buildDockerfile('pybk', 'pybk/Dockerfile', env.DOCKER_HUB_REPO_PYBK, version, changedFiles) // DEPRECATED
                    buildDockerfile('simple', 'simple/Dockerfile', env.DOCKER_HUB_REPO_SIMPLE, version, changedFiles)
                    buildDockerfile('pyannote', 'pyannote/Dockerfile', env.DOCKER_HUB_REPO_PYANNOTE, version, changedFiles)
                }
            }
        }
    }
}