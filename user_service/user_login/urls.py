
from django.urls import path, include
from user_model import views as modelViews
from user_login import views as loginViews
from user_info import views as infoViews
urlpatterns = [
    # path('userlogin', views.user_login),
    path('userregistration', modelViews.registration_req),
    path('login', loginViews.user_login),
    path('userInfo', infoViews.user_info),
]