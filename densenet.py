import torch
import torch.nn as nn
import torchvision.models as models
import torchaudio as taudio
import torchvision as tvision

class DenseNet(nn.Module):
    def __init__(self, num_of_classes, pretrained=True, multi_spectral = False):
        super(DenseNet, self).__init__()
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_of_classes)

        self.resizing_transform = tvision.transforms.Resize(size = (224,224), interpolation = tvision.transforms.InterpolationMode.NEAREST,
                              antialias = False)

        if multi_spectral:
            self.mel_spectrogram_transform = [taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 1024, hop_length = 2205, n_mels = 32, f_max = 16000),
                                         taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 2048, hop_length = 2205, n_mels = 32, f_max = 16000),
                                         taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 256, hop_length = 2205, n_mels = 32, f_max = 16000)]
        else:
            self.mel_spectrogram_transform = [taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 2048, hop_length = 2205, n_mels = 32, f_max = 16000),
                                         taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 2048, hop_length = 2205, n_mels = 32, f_max = 16000),
                                         taudio.transforms.MelSpectrogram(sample_rate = 44100, n_fft = 2048, hop_length = 2205, n_mels = 32, f_max = 16000)]

    def forward(self, audio):
        """
        Leverage the GPU for spectrogram generation
        """

        spectrogram_image_like = self.mel_spectrogram_transform[0](audio)
        spectrogram_image_like = torch.unsqueeze(spectrogram_image_like, dim = 0)

        for transform in self.mel_spectrogram_transform[1::]:
            other_resolution_mel_spec = transform(audio)
            other_resolution_mel_spec = torch.unsqueeze(other_resolution_mel_spec, dim = 0)
            spectrogram_image_like = torch.cat(tensors = (spectrogram_image_like, other_resolution_mel_spec), dim = 0)
    
        spectrogram_image_like = self.resizing_transform(spectrogram_image_like)
        spectrogram_image_like = torch.unsqueeze(spectrogram_image_like, dim = 0)
        
        output = self.model(spectrogram_image_like)
        return output
    
    def freeze_up_to_layer(self, layer_no):
        pass