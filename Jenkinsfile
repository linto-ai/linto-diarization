def notifyLintoDeploy(service_name, tag, commit_sha) {
    echo "Notifying linto-deploy for ${service_name}:${tag} (commit: ${commit_sha})..."
    withCredentials([usernamePassword(
        credentialsId: 'linto-deploy-bot',
        usernameVariable: 'GITHUB_APP',
        passwordVariable: 'GITHUB_TOKEN'
    )]) {
        writeFile file: 'payload.json', text: "{\"event_type\":\"update-service\",\"client_payload\":{\"service\":\"${service_name}\",\"tag\":\"${tag}\",\"commit_sha\":\"${commit_sha}\"}}"
        sh 'curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" -d @payload.json https://api.github.com/repos/linto-ai/linto-deploy/dispatches'
    }
}

def buildDockerfile(main_folder, dockerfilePath, image_name, version, changedFiles, commit_sha) {
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

            // Notify linto-deploy after successful push (only for master branch)
            if (version != 'latest-unstable') {
                def service_name = image_name.replace('lintoai/', '')
                notifyLintoDeploy(service_name, version, commit_sha)
            }
        }
    }
}

// Best-effort deploy of a freshly built image to the staging cluster (full CI/CD).
// Needs a Jenkins SSH credential 'staging-deploy-ssh' (key for ubuntu@bm2-3s);
// if absent the build still succeeds (push-only).
def stagingDeploy(image_name, tag) {
    try {
        withCredentials([sshUserPrivateKey(credentialsId: 'staging-deploy-ssh', keyFileVariable: 'SSH_KEY', usernameVariable: 'SSH_USER')]) {
            sh "ssh -i \$SSH_KEY -o StrictHostKeyChecking=no \$SSH_USER@163.114.159.33 'staging-deploy ${image_name} ${tag}'"
        }
    } catch (err) {
        echo "Staging auto-deploy skipped for ${image_name}:${tag} (add the 'staging-deploy-ssh' credential to enable): ${err}"
    }
}

pipeline {
    agent any
    environment {
        // DOCKER_HUB_REPO_PYBK   = "lintoai/linto-diarization-pybk" // DEPRECATED
        DOCKER_HUB_REPO_PYANNOTE = "lintoai/linto-diarization-pyannote"
        DOCKER_HUB_REPO_SIMPLE = "lintoai/linto-diarization-simple"
        STAGING_REGISTRY_PYANNOTE = "registry.staging.linto.ai/lintoai/linto-diarization-pyannote"
        STAGING_REGISTRY_CRED = 'staging-registry-credentials'
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
                    def commit_sha = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
                    echo "My changed files: ${changedFiles}"

                    // // DEPRECATED
                    // version = sh(
                    //     returnStdout: true,
                    //     script: "awk -v RS='' '/#/ {print; exit}' pybk/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    // ).trim()
                    // buildDockerfile('pybk', 'pybk/Dockerfile', env.DOCKER_HUB_REPO_PYBK, version, changedFiles, commit_sha)

                    version = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' simple/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()
                    buildDockerfile('simple', 'simple/Dockerfile', env.DOCKER_HUB_REPO_SIMPLE, version, changedFiles, commit_sha)

                    version = sh(
                        returnStdout: true,
                        script: "awk -v RS='' '/#/ {print; exit}' pyannote/RELEASE.md | head -1 | sed 's/#//' | sed 's/ //'"
                    ).trim()
                    buildDockerfile('pyannote', 'pyannote/Dockerfile', env.DOCKER_HUB_REPO_PYANNOTE, version, changedFiles, commit_sha)
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

                    // buildDockerfile('pybk', 'pybk/Dockerfile', env.DOCKER_HUB_REPO_PYBK, version, changedFiles, '') // DEPRECATED
                    buildDockerfile('simple', 'simple/Dockerfile', env.DOCKER_HUB_REPO_SIMPLE, version, changedFiles, '')
                    buildDockerfile('pyannote', 'pyannote/Dockerfile', env.DOCKER_HUB_REPO_PYANNOTE, version, changedFiles, '')
                }
            }
        }

        stage('Docker build for staging branches') {
            when {
                branch 'staging/*'
            }
            steps {
                echo 'Building staging feature-branch image (pyannote, private registry, never Docker Hub)'
                script {
                    def slug = env.BRANCH_NAME.replaceFirst('^staging/', '').replaceAll('[^a-zA-Z0-9]+', '-').toLowerCase()
                    def tag = "dev-${slug}"
                    def image = docker.build(env.STAGING_REGISTRY_PYANNOTE, "-f pyannote/Dockerfile .")
                    docker.withRegistry('https://registry.staging.linto.ai', env.STAGING_REGISTRY_CRED) {
                        image.push(tag)
                    }
                    stagingDeploy('linto-diarization-pyannote', tag)
                }
            }
        }
    }
}
