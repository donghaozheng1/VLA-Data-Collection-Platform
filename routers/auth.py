from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# ===== Demo 用户表（内存态，重启即还原，不持久化） =====
USERS = {
    "admin": {"username": "admin", "password": "admin123", "role": "管理员"},
}

# 会话 Cookie 名（仅记录用户名，Demo 不做签名）
SESSION_COOKIE = "hkc_user"


def current_user(request: Request):
    """从 Cookie 中取当前登录用户，未登录返回 None"""
    username = request.cookies.get(SESSION_COOKIE)
    if not username:
        return None
    return USERS.get(username)


@router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"request": request})


@router.post("/login")
async def login(request: Request, username: str = Form(""), password: str = Form("")):
    user = USERS.get(username.strip())
    if user is None or user["password"] != password:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"request": request, "error": "用户名或密码错误"},
            status_code=401,
        )
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(SESSION_COOKIE, user["username"], httponly=True)
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@router.get("/settings")
async def settings_page(request: Request):
    user = current_user(request) or USERS["admin"]
    return templates.TemplateResponse(request, "settings.html", {"request": request, "user": user})


@router.post("/settings/change-password")
async def change_password(
    request: Request,
    old_password: str = Form(""),
    new_password: str = Form(""),
):
    user = current_user(request)
    context = {"request": request, "user": user or USERS["admin"]}

    if user is None:
        context["error"] = "登录状态已失效，请重新登录"
        return templates.TemplateResponse(request, "settings.html", context, status_code=401)
    if user["password"] != old_password:
        context["error"] = "当前密码不正确"
        return templates.TemplateResponse(request, "settings.html", context, status_code=400)
    if len(new_password) < 6:
        context["error"] = "新密码长度至少 6 位"
        return templates.TemplateResponse(request, "settings.html", context, status_code=400)

    user["password"] = new_password
    context["success"] = "密码修改成功"
    return templates.TemplateResponse(request, "settings.html", context)
