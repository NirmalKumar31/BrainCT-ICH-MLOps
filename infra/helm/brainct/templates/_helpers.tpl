{{- define "brainct.name" -}}{{ .Chart.Name }}{{- end -}}
{{- define "brainct.fullname" -}}{{ printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}{{- end -}}
{{- define "brainct.labels" -}}
app.kubernetes.io/name: {{ include "brainct.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
{{- define "brainct.selector" -}}
app.kubernetes.io/name: {{ include "brainct.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
