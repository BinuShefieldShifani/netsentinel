// Jenkins declarative pipeline. Every stage calls ci/run_checks.sh, the same
// script GitHub Actions runs, so both CI systems check exactly the same things.
//
// For change-based review (Gerrit), point the Gerrit Trigger plugin at this
// job: it builds each uploaded patch set and votes Verified +1 / -1 from the
// result. See CONTRIBUTING.md.

pipeline {
    agent {
        dockerfile {
            filename 'ci/Dockerfile'
        }
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    environment {
        PIP_CACHE_DIR = "${WORKSPACE}/.pip-cache"
        REPORTS_DIR = 'reports/ci'
        NETSENTINEL_REQUIRE_CPP = '1'   // parity tests must run, not skip
        NETSENTINEL_LLM = 'none'        // CI never calls an LLM
    }

    stages {
        stage('Setup') {
            steps {
                sh 'python -m venv .venv && . .venv/bin/activate && ci/run_checks.sh setup'
            }
        }
        stage('Lint') {
            steps {
                sh '. .venv/bin/activate && ci/run_checks.sh lint'
            }
        }
        stage('C++ build and GoogleTest') {
            steps {
                sh '. .venv/bin/activate && ci/run_checks.sh cpp'
            }
        }
        stage('Python tests') {
            steps {
                sh '. .venv/bin/activate && ci/run_checks.sh test'
            }
        }
        stage('Train and evaluate') {
            steps {
                sh '. .venv/bin/activate && ci/run_checks.sh train'
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/ci/*.xml'
            archiveArtifacts artifacts: 'reports/ci/**', allowEmptyArchive: true, fingerprint: true
        }
    }
}
