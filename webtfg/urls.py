"""webtfg URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

from deepleasy.views import ModelBuilderSupervised, ModelOptions, ModelProgress, ModelHistory, ModelGetter, \
    ModelBuilderUnsupervised

urlpatterns = [
    path('admin/', admin.site.urls),

    path('api/model/options/', ModelOptions.as_view()),
    path('api/model/builder/supervised', ModelBuilderSupervised.as_view()),
    path('api/model/builder/unsupervised', ModelBuilderUnsupervised.as_view()),

    path('api/model/progress/', ModelProgress.as_view()),
    path('api/model/history/', ModelHistory.as_view()),
    path('api/model/', ModelGetter.as_view()),

    path('api/auth/token/', TokenObtainPairView.as_view()),
    path('api/auth/verify/', TokenVerifyView.as_view()),
    path('api/auth/refresh/', TokenRefreshView.as_view()),
]

"""
    path('login/', login_view, name="login"),
    path('logout/', logout_view, name="logout"),
    path('', home_view),

    path('tool/', tool_view, name="tool"),
    path('analysing/', analysing_test, name="analysing"),
"""
