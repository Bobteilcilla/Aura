build_container_local:
	docker build --platform linux/amd64 --tag=${IMAGE}:dev .

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
		-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/aura-477808-9705dd39a66e.json \
		-e MODEL_BUCKET=${MODEL_BUCKET} \
		-v $(PWD)/gcp/aura-477808-9705dd39a66e.json:/secrets/aura-477808-9705dd39a66e.json:ro \
		-p 8080:8000 \
		${IMAGE}:dev
