build_container_local:
	docker build --tag=${IMAGE}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 ${IMAGE}:dev

build_for_production:
	docker build \
		--platform linux/amd64 \
    -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${IMAGE}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION}

run_container_gcp:
	docker run -it \
		-e PORT=8000 \
		-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/aura-477808-dd52886578d3.json \
		-v $(PWD)/gcp/aura-477808-dd52886578d3.json:/secrets/aura-477808-dd52886578d3.json:ro \
		-p 8080:8000 \
		${IMAGE}:dev
