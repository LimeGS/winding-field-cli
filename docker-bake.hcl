group "default" {
  targets = ["cpu", "cuda"]
}

target "common" {
  context = "."
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
}

target "cpu" {
  inherits = ["common"]
  tags = ["winding-field:cpu"]
}

target "cuda" {
  inherits = ["common"]
  tags = ["winding-field:cuda"]
  args = {
    BASE_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
    TORCH_VARIANT = "cu128"
    NVIDIA_REQUIRE_CUDA = "cuda>=12.8"
  }
}
