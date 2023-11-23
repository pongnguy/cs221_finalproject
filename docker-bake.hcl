variable "TAG" {
  default = "jupyter"
}

group "default" {
  targets = ["cs221"]
}

target "cs221" {
  dockerfile = "Dockerfile"
  tags = ["alfred/cs221:${TAG}"]
  output = ["type=docker"]
}

